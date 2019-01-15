"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import tensorflow as tf
from time import gmtime, strftime
# from osgeo import gdal
# import tifffile as tiff
from PIL import Image
import glob
from skimage.transform import resize
from skimage import exposure
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
import keras
import collections  
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.externals import joblib
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def balance_data(data, labels, index=None, samples_per_class=32):

    if (index is None):
        idxs = np.arange(len(labels))
    else:
        idxs = index

    classes = np.unique(labels)
    # print (classes)
    num_total_samples = len(classes) * samples_per_class
    out_labels = np.zeros((num_total_samples), dtype='uint8')
    out_idxs = np.zeros((num_total_samples), dtype='uint32')
    out_data = np.zeros((num_total_samples, data.shape[1]), dtype='float32')

    k = 0
    for clss in classes:
        clss_data = data[labels == clss]
        clss_index = idxs[labels == clss]
        clss_labels = labels[labels == clss]
        num_samples = len(clss_labels)
        # print(num_samples)
        if num_samples >= samples_per_class:
            # Choose samples randomly
            index = range(len(clss_labels))
            index = np.random.choice(index, samples_per_class, replace=False)
            out_labels[k*samples_per_class:(k+1)*samples_per_class] = clss_labels[index]
            out_data[k*samples_per_class:(k+1)*samples_per_class] = clss_data[index]
            out_idxs[k*samples_per_class:(k+1)*samples_per_class] = clss_index[index]
            # print(clss_index[index])

        else:
            # do oversampling
            index = range(len(clss_labels))
            index = np.random.choice(index, samples_per_class, replace=True)
            out_labels[k*samples_per_class:(k+1)*samples_per_class] = clss_labels[index]
            out_data[k*samples_per_class:(k+1)*samples_per_class] = clss_data[index]
            out_idxs[k*samples_per_class:(k+1)*samples_per_class] = clss_index[index]
        k += 1
    # Permute samples randomly
    idx = np.random.permutation(num_total_samples)
    out_data = out_data[idx]
    out_labels = out_labels[idx]
    out_idxs = out_idxs[idx]

    return out_data, out_labels, out_idxs


# -----------------------------
# new added functions for pix2pix
def load_landsat(path):
    images = sorted(glob.glob(path + '*.tif'))
    band = load_tiff_image(images[0])
    rows, cols = band.shape
    img = np.zeros((rows, cols, 7), dtype='float32')
    num_band = 0
    for im in images:
        if 'B8' not in im and 'QB' not in im:
            band = load_tiff_image(im)
            img[:, :, num_band] = band
            num_band += 1
        if 'QB' in im:
            cloud_mask = load_tiff_image(im)
            cloud_mask[cloud_mask != 0] = 1
    return img, cloud_mask


def load_sentinel2(path):
    images = sorted(glob.glob(path + '*.tif'))
    band = load_tiff_image(images[0])
    rows, cols = band.shape
    img = np.zeros((rows, cols, 4), dtype='float32')
    num_band = 0
    for im in images:
        if 'B02' in im or 'B03' in im or 'B04' in im or 'B08' in im:
            band = load_tiff_image(im)
            img[:, :, num_band] = band
            num_band += 1
    return img


def load_sar(path):
    sar = np.load(path).astype('float32')
#    sar = np.rollaxis(sar, 0, 3)
    sar = np.float32(sar)
    return sar


def load_tiff_image(patch):
    # Read Mask Image
    print patch
    img = Image.open(patch)
    img = np.array(img)
    # img = tiff.imread(patch)
    # gdal_header = gdal.Open(patch)
    # img = gdal_header.ReadAsArray()
    return img


def load_mask(mask_path=None):
    if mask_path is None:
        mask_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/New_Masks/TrainTestMasks/TrainTestMask_50_50_Dec.tif'
    mask = load_tiff_image(mask_path)
    return mask


def load_images(mask_path=None,
                sar_path=None,
                opt_path=None):

    if sar_path is None and opt_path is None:
        print 'Image patches were not defined  !!!'
        return False

    mask = load_mask(mask_path)
    sar = load_sar(sar_path)
    opt, cloud_mask = load_landsat(opt_path)

    return mask, sar, opt, cloud_mask


# def resampler(img, dtype=None):
#     if dtype is None:
#         print("Define data type image")
#         return 0
#     img_shape = img.shape
#     if img.ndim == 3:
#         im_c = np.zeros((img_shape[0]+1, img_shape[1], img_shape[2]), dtype=img.dtype)
#         im_c[0:-1] = img
#         im = np.zeros((int(im_c.shape[0]/3), int(im_c.shape[1]/3), img_shape[2]), dtype=img.dtype)
#         for i in range(img.shape[2]):
#             im[:, :, i] = scipy.misc.imresize(im_c[:, :, i], (im.shape[0], im.shape[1]), mode='F')
# #        im = resize(im, (im.shape[0]/3, im.shape[1]/3), preserve_range=True)
#     else:
#         im = np.zeros((img_shape[0]+1, img_shape[1]), dtype=img.dtype)
#         im[0:-1] = img
#         im = resize(np.uint8(im), (im.shape[0]//3, im.shape[1]//3), order=0, preserve_range=True)
#         # im = scipy.misc.imresize(im, (int(im.shape[0]/3), int(im.shape[1]/3)), interp ='nearest', mode='F')
#     return im

def resampler(img, dtype=None):
    if dtype is None:
        print("Define data type image")
        return 0
    img_shape = img.shape
    print (img_shape)
    if dtype is 'float32':
        im = np.zeros((img_shape[0]+1, img_shape[1], img_shape[2]), dtype=img.dtype)
        im[0:-1] = img
        # im = np.zeros((int(im_c.shape[0]/3), int(im_c.shape[1]/3), img_shape[2]), dtype=img.dtype)
        # for i in range(img.shape[2]):
            # im[:, :, i] = scipy.misc.imresize(im_c[:, :, i], (im.shape[0], im.shape[1]), mode='F')
        im = resize(im, (im.shape[0]//3, im.shape[1]//3), order=1, preserve_range=True)
        print im.shape
    elif dtype is 'uint8':
        im = np.zeros((img_shape[0]+1, img_shape[1]), dtype=img.dtype)
        im[0:-1] = img
        im = resize(np.uint8(im), (im.shape[0]//3, im.shape[1]//3), order=0, preserve_range=True)
    else:
        print("Define data type image")
    return im


def up_sampler(im, dtype=None):
    if dtype is None:
        print("Define data type image")
        return 0
    img_shape = im.shape
    print (img_shape)
    if dtype is 'float32':
        im = resize(im, (3 * im.shape[0], 3 * im.shape[1]), order=1, preserve_range=True)
        im = im[:-1]
        print im.shape
    elif dtype is 'uint8':
        im = resize(np.uint8(im), (3 * im.shape[0], 3 * im.shape[1]), order=0, preserve_range=True)
        im = im[:-1]
    else:
        print("Define data type image")
    return im

#def get_patch(img, ksize, row, col):
##    _, cols, _ = img.shape
##    print row, col
#    if ksize % 2 != 0:
#        patch = img[row-int(ksize/2):row+int(ksize/2)+1, col-int(ksize/2):col+int(ksize/2)+1, :]
#    else:
#        patch = img[row-int(ksize/2):row+int(ksize/2), col-int(ksize/2):col+int(ksize/2), :]
#    return patch

def get_patch(img, ksize, row, col):
#    _, cols, _ = img.shape
#    print row, col
    if ksize % 2 != 0:
        patch = img[row-int(ksize/2):row+int(ksize/2)+1, col:col+int(ksize)+1, :]
    else:
        patch = img[row-int(ksize/2):row+int(ksize/2), col:col+int(ksize), :]
    return patch

def load_data4Classifier_quemadas(self, samples_list, sample_index=None, samples_per_class=None):

    if sample_index is None:
        idxc = np.random.randint(0, len(samples_list))
    else:
        idxc = sample_index

    data_Dic = np.load(samples_list[idxc]).item()
    labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
    sar_patch = np.array(data_Dic['sar']).astype('float32')
    opt_patch = np.array(data_Dic['opt']).astype('float32')
    # labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
    trn_samples = opt_patch.copy().reshape(len(labels_patch), self.n_features)
    trn_labels = labels_patch.copy()
    if samples_per_class is None:
        n_samples_quemadas = np.sum(trn_labels == 1)
    else:
        n_samples_quemadas = samples_per_class
    # print("num samples quemadas ---> ", n_samples_quemadas)
    # Balance data
    trn_samples, trn_labels, idx_samples = balance_data(trn_samples, trn_labels, samples_per_class=n_samples_quemadas)
    # print(trn_labels)
    # print(labels_patch[idx_samples])
    # print(idx_samples)
    # print(np.sum(labels_patch[idx_samples] - trn_labels))

    return sar_patch, opt_patch, trn_samples, trn_labels.reshape(len(trn_labels), 1), labels_patch, idx_samples


def load_data4Validation_quemadas(self, samples_list, balance_data=False):
    val_Data = np.zeros((256**2 * len(samples_list), self.n_features))
    val_Labels = np.zeros((256**2 * len(samples_list)))
    index = np.arange(0, 256**2 * len(samples_list))
    for idxc in range(0, len(samples_list)):
        data_Dic = np.load(samples_list[idxc]).item()
        labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
        opt_patch = np.array(data_Dic['opt']).astype('float32').reshape(len(labels_patch), self.n_features)
        labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
        val_Data[idxc*256**2:(idxc+1)*256**2] = opt_patch
        val_Labels[idxc*256**2:(idxc+1)*256**2] = labels_patch

    n_samples_quemadas = np.sum(val_Labels == 1)
    # print(n_samples_quemadas)

    # Balance data
    if balance_data is True:
        val_Data, val_Labels, _ = balance_data(val_Data, val_Labels, samples_per_class=n_samples_quemadas)
    return val_Data, val_Labels.reshape(len(val_Labels), 1)


def load_data4Classifier(self, samples_list, sample_index=None, samples_per_class=None):

    if sample_index is None:
        idxc = np.random.randint(0, len(samples_list))
    else:
        idxc = sample_index

    data_Dic = np.load(samples_list[idxc]).item()
    labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
    sar_patch = np.array(data_Dic['sar']).astype('float32')
    opt_patch = np.array(data_Dic['opt']).astype('float32')
    # labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
    trn_samples = opt_patch.copy().reshape(len(labels_patch), self.n_features)
    trn_labels = labels_patch.copy()
    if samples_per_class is None:
        n_samples_quemadas = np.sum(trn_labels == 1)
    else:
        n_samples_quemadas = samples_per_class
    # print("num samples quemadas ---> ", n_samples_quemadas)
    # Balance data
    trn_samples, trn_labels, idx_samples = balance_data(trn_samples, trn_labels, samples_per_class=n_samples_quemadas)
    # print(trn_labels)
    # print(labels_patch[idx_samples])
    # print(idx_samples)
    # print(np.sum(labels_patch[idx_samples] - trn_labels))

    return sar_patch, opt_patch, trn_samples, trn_labels.reshape(len(trn_labels), 1), labels_patch, idx_samples


def load_dataCV(self, samples_list, sample_index=None):
    if sample_index is None:
        idxc = np.random.randint(0, len(samples_list))
    else:
        idxc = sample_index
    data_Dic = np.load(samples_list[idxc]).item()
    sar_patch = np.array(data_Dic['sar']).astype('float32')
    opt_patch = np.array(data_Dic['opt']).astype('float32')
    batch_images = np.concatenate((sar_patch.reshape(1, self.image_size,
                                                     self.image_size,
                                                     self.input_c_dim),
                                   opt_patch.reshape(1, self.image_size,
                                                     self.image_size,
                                                     self.output_c_dim)),
                                  axis=3)

    return batch_images

def load_data4ClassifierCV_one_hot(self, samples_list, labels2new_labels=None, sample_index=None, samples_per_class=32):
    if sample_index is None:
        idxc = np.random.randint(0, len(samples_list))
    else:
        idxc = sample_index
    data_Dic = np.load(samples_list[idxc]).item()
    # labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
    # sar_patch = np.array(data_Dic['sar']).astype('float32')
    # opt_patch = np.array(data_Dic['opt']).astype('float32').reshape(len(labels_patch), self.n_features)
    labels_patch = np.array(data_Dic['labels']).astype('float32')
    sar_patch = np.array(data_Dic['sar']).astype('float32')
    opt_patch = np.array(data_Dic['opt']).astype('float32')
    img_SO = np.concatenate((sar_patch, opt_patch), axis=2)
    # print(sar_patch.max(), sar_patch.min())
    # print(opt_patch.max(), opt_patch.min())
    img_SO, labels_patch = preprocess_S_and_O_and_L(img_SO, labels_patch)
    labels_patch = labels_patch.ravel()
    sar_patch = img_SO[:, :, :self.input_c_dim]
    opt_patch = img_SO[:, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
    trn_samples = opt_patch.copy().reshape(len(labels_patch), self.n_features)

    if labels2new_labels is None:
        return sar_patch, opt_patch

    index = np.arange(0, len(labels_patch))
    # remove background
    # print (samples_list[idxc])
    # print (np.unique(labels_patch))
    index = index[labels_patch != 0]
    if len(index) is 0:
        # generate dummy data
        beta = 1.0
        trn_samples = trn_samples[:10]
        trn_labels_one_hot = np.zeros((10, self.n_classes))
        return sar_patch, opt_patch, trn_samples, trn_labels_one_hot, labels_patch, np.ones(10), beta
    beta = 1.0
    trn_samples = trn_samples[labels_patch != 0]
    trn_labels = labels_patch.copy()[labels_patch != 0]
    index = index[trn_labels != 5]
    trn_samples = trn_samples[trn_labels != 5]
    trn_labels = trn_labels[trn_labels != 5]
    index = index[trn_labels != 10]
    trn_samples = trn_samples[trn_labels != 10]
    trn_labels = trn_labels[trn_labels != 10]
    # print (np.unique(trn_labels))
    # Balance data
    trn_samples, trn_labels, idx_samples = balance_data(trn_samples, trn_labels, index=index, samples_per_class=samples_per_class)
    # convert labels index and also one-hot encoding.
    trn_labels_one_hot = np.zeros((len(trn_labels), self.n_classes), dtype='float32')
    for idx_l in range(len(trn_labels)):
        one_hot = labels2new_labels[trn_labels[idx_l]]
        trn_labels_one_hot[idx_l, one_hot] = 1
    return sar_patch, opt_patch, trn_samples, trn_labels_one_hot, labels_patch, idx_samples, beta


def convert_labels(labels, labels2new_labels):
    mask_classes2_zero = (labels==5)
    labels[mask_classes2_zero] = 0
    new_labels = -1 * np.ones_like(labels)
    classes = np.unique(labels)
    for clas in classes:
        if clas != 0:
            new_labels[labels == clas] = labels2new_labels[clas]
    return new_labels


def load_data_Dic(samples_list,
                  idxc,
                  load_size=286,
                  fine_size=256,
                  random_transformation=False,
                  multitemporal=False,
                  labels=False):
    data_Dic = np.load(samples_list[idxc]).item()
    if (multitemporal):
        sar_t0 = np.array(data_Dic['sar_t0']).astype('float32')
        opt_t0 = np.array(data_Dic['opt_t0']).astype('float32')
        sar_t1 = np.array(data_Dic['sar_t1']).astype('float32')
        opt_t1 = np.array(data_Dic['opt_t1']).astype('float32')
        if labels:
            labels = np.array(data_Dic['labels']).astype('uint8')
        sar_t0, opt_t0, sar_t1, opt_t1, labels =  tranformations(sar_t0,
                                                                 opt_t0,
                                                                 sar_t1,
                                                                 opt_t1,
                                                                 load_size=load_size,
                                                                 fine_size=fine_size,
                                                                 random_transformation=False,
                                                                 labels=labels)
        return sar_t0, opt_t0, sar_t1, opt_t1, labels
    else:
        sar_t0 = np.array(data_Dic['sar_t0']).astype('float32')
        opt_t0 = np.array(data_Dic['opt_t0']).astype('float32')
        if labels:
            labels = np.array(data_Dic['labels']).astype('uint8')
        sar_t0, opt_t0, labels = tranformations(sar_t0,
                                                opt_t0,
                                                load_size=load_size,
                                                fine_size=fine_size,
                                                random_transformation=False,
                                                labels=labels)
    return sar_t0, opt_t0, labels


def load_data_Dic_Multiresolution(samples_list,
                                  idxc,
                                  load_size=286,
                                  fine_size=256,
                                  random_transformation=False,
                                  multitemporal=False,
                                  labels=False,
                                  labels2new_labels=None):
    data_Dic = np.load(samples_list[idxc]).item()
    if (multitemporal):
        sar_t0 = np.array(data_Dic['sar_t0']).astype('float32')
        opt_t0 = np.array(data_Dic['opt_t0']).astype('float32')
        sar_t1 = np.array(data_Dic['sar_t1']).astype('float32')
        opt_t1 = np.array(data_Dic['opt_t1']).astype('float32')
        if labels:
            labels = np.array(data_Dic['labels']).astype('uint8')
            labels = convert_labels(labels, labels2new_labels)
        sar_t0, opt_t0, sar_t1, opt_t1, labels =  transformations_multiresolution(sar_t0,
                                                                                  opt_t0,
                                                                                  sar_t1,
                                                                                  opt_t1,
                                                                                  load_size=load_size,
                                                                                  fine_size=fine_size,
                                                                                  random_transformation=False,
                                                                                  labels=labels)
        return sar_t0, opt_t0, sar_t1, opt_t1, labels
    else:
        sar_t0 = np.array(data_Dic['sar_t0']).astype('float32')
        opt_t0 = np.array(data_Dic['opt_t0']).astype('float32')
        if labels:
            labels = np.array(data_Dic['labels']).astype('uint8')
            labels = convert_labels(labels, labels2new_labels)
        sar_t0, opt_t0, labels = transformations_multiresolution(sar_t0,
                                                                 opt_t0,
                                                                 load_size=load_size,
                                                                 fine_size=fine_size,
                                                                 random_transformation=False,
                                                                 labels=labels)
        return sar_t0, opt_t0, labels


def load_data4ClassifierCV(self, samples_list, labels2new_labels=None, sample_index=None, samples_per_class=32):
    if sample_index is None:
        idxc = np.random.randint(0, len(samples_list))
    else:
        idxc = sample_index
    data_Dic = np.load(samples_list[idxc]).item()
    # labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
    # sar_patch = np.array(data_Dic['sar']).astype('float32')
    # opt_patch = np.array(data_Dic['opt']).astype('float32').reshape(len(labels_patch), self.n_features)
    labels_patch = np.array(data_Dic['labels']).astype('uint8')
    img_A = np.array(data_Dic['img_A']).astype('float32')
    img_B = np.array(data_Dic['img_B']).astype('float32')
    img_SO = np.concatenate((img_A, img_B), axis=2)
    # print(sar_patch.max(), sar_patch.min())
    # print(opt_patch.max(), opt_patch.min())
    img_SO, labels_patch = preprocess_S_and_O_and_L(img_SO, labels_patch)
    labels_patch = labels_patch.ravel()
    img_A = img_SO[:, :, :self.input_c_dim]
    img_B = img_SO[:, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
    trn_samples = img_B.copy().reshape(len(labels_patch), self.n_features)

    if labels2new_labels is None:
        return img_A, img_B

    index = np.arange(0, len(labels_patch))
    # remove background
    # index = index[labels_patch != 0]
    # print ("number of labeled pixels --->", len(index))
    num_labels = sum(labels_patch != 0)
    if num_labels == 0:
        # generate dummy data
        beta = 0.0
        trn_samples = trn_samples[:10]
        trn_labels = np.zeros((10))
        return img_A, img_B, trn_samples, trn_labels, labels_patch, np.ones(10), beta
    beta = 1.0
    trn_labels = labels_patch.copy()
    eli_classes = (trn_labels != 0) * (trn_labels != 5) * (trn_labels != 10)
    trn_labels = trn_labels[eli_classes]
    trn_samples = trn_samples[eli_classes]
    index = index[eli_classes]
    # index = index[trn_labels != 5]
    # trn_samples = trn_samples[trn_labels != 5]
    # trn_labels = trn_labels[trn_labels != 5]
    # index = index[trn_labels != 10]
    # trn_samples = trn_samples[trn_labels != 10]
    # trn_labels = trn_labels[trn_labels != 10]
    # min_classes = (trn_labels==4) + (trn_labels==9) + (trn_labels==11)
    # trn_labels[~ min_classes] = 0
    # print (np.unique(trn_labels))
    # samples_per_class = len(trn_labels)//len(np.unique(trn_labels))
    # Balance data
    trn_samples, trn_labels, idx_samples = balance_data(trn_samples, trn_labels, index=index, samples_per_class=samples_per_class)
    new_labels = np.zeros_like(trn_labels)
    classes = np.unique(trn_labels)
    for clas in classes:
        new_labels[trn_labels == clas] = labels2new_labels[clas]

    return img_A, img_B, trn_samples, new_labels, labels_patch, idx_samples, beta


def load_data4ClassifierCV_oneClass(self, samples_list, labels2new_labels=None, sample_index=None, samples_per_class=32):
    if sample_index is None:
        idxc = np.random.randint(0, len(samples_list))
    else:
        idxc = sample_index
    data_Dic = np.load(samples_list[idxc]).item()
    labels_patch = np.array(data_Dic['labels']).astype('uint8')
    img_A = np.array(data_Dic['img_A']).astype('float32')
    img_B = np.array(data_Dic['img_B']).astype('float32')
    img_SO = np.concatenate((img_A, img_B), axis=2)
    # print(sar_patch.max(), sar_patch.min())
    # print(opt_patch.max(), opt_patch.min())
    img_SO, labels_patch = preprocess_S_and_O_and_L(img_SO, labels_patch)
    labels_patch = labels_patch.ravel()
    img_A = img_SO[:, :, :self.input_c_dim]
    img_B = img_SO[:, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
    trn_samples = img_B.copy().reshape(len(labels_patch), self.n_features)

    if labels2new_labels is None:
        return img_A, img_B

    index = np.arange(0, len(labels_patch))
    # remove background
    # index = index[labels_patch != 0]
    # print ("number of labeled pixels --->", len(index))
    num_labels = sum(labels_patch != 0)
    if num_labels == 0:
        # generate dummy data
        beta = 0.0
        trn_samples = trn_samples[:10]
        trn_labels = np.zeros((10))
        return img_A, img_B, trn_samples, trn_labels, labels_patch, np.ones(10), beta
    beta = 1.0
    trn_labels = labels_patch.copy()
    eli_classes = (trn_labels != 0)
    trn_labels = trn_labels[eli_classes]
    trn_samples = trn_samples[eli_classes]
    index = index[eli_classes]
    trn_labels[trn_labels != 2] = 0
    # samples_per_class = len(trn_labels)//len(np.unique(trn_labels))
    # Balance data
    trn_samples, trn_labels, idx_samples = balance_data(trn_samples, trn_labels, index=index, samples_per_class=samples_per_class)
    new_labels = np.zeros_like(trn_labels)
    classes = np.unique(trn_labels)
    for clas in classes:
        new_labels[trn_labels == clas] = labels2new_labels[clas]

    return img_A, img_B, trn_samples, new_labels, labels_patch, idx_samples, beta


    # 2.0: 117728, 3.0: 163729, 4.0: 513, 6.0: 17758, 7.0: 30870, 8.0: 4265, 9.0: 1764, 11.0: 5983,   
    # 3.0: 163729, 2.0: 117728, 7.0: 30870, 6.0: 17758, 11.0: 5983, 8.0: 4265, 9.0: 1764, 4.0: 513
    # [0.73838015, 1.0682159, 2.40680794, 2.95975666, 4.04767066, 4.38615059, 5.26900882, 6.50407221]
    # [1.0682159, 0.73838015, 6.50407221, 2.95975666, 2.40680794, 4.38615059, 5.26900882, 4.04767066]
    # [2: 0.36377285,  3: 0.2615679 , 4: 83.48196881,  6: 2.41165953,  7: 1.38730969, 8: 10.04132474, 9: 24.2779195 ,  11: 7.1579893 ]
    # Counter({3.0: 163729, 2.0: 117728, 7.0: 30870, 6.0: 17758, 11.0: 5983, 8.0: 4265, 9.0: 1764, 4.0: 513, 10.0: 108})
    # Counter({2.0: 117728, 3.0: 163729, 4.0: 513, 6.0: 17758, 7.0: 30870, 8.0: 4265, 9.0: 1764, 10.0: 108, 11.0: 5983})

def cal_loss(self, logits, labels):
    freq = [1.17728e+05, 1.63729e+05, 5.13000e+02, 1.77580e+04, 3.08700e+04, 4.26500e+03, 1.76400e+03, 1.08000e+02, 5.98300e+03]
    # loss_weight1 = np.mean(freq)/freq
    # loss_weight1[loss_weight1<1.0] = 1.0
    max_freq = np.max(freq)
    loss_weight = np.array([
        max_freq / 117728.0,
        max_freq / 163729.0,
        max_freq / 513.0,
        max_freq / 17758.0,
        max_freq / 30870.0,
        max_freq / 4265.0,
        max_freq / 1764.0,
        max_freq / 108.0,
        max_freq / 5983.0
    ])
    loss_weight[loss_weight>100] = 100

    print ('weight classes --->', loss_weight)

    labels = tf.to_int64(labels)
    loss, accuracy, prediction = weighted_loss(self, logits, labels, number_class=self.n_classes, frequency=loss_weight)
    return loss, accuracy, prediction


def cal_loss_Q(self, logits, labels):
    
    loss_weight = np.array([
        1,
        100
    ])
    
    print ('weight classes --->', loss_weight)

    labels = tf.to_int64(labels)
    loss, accuracy, prediction = weighted_loss(self, logits, labels, number_class=self.n_classes, frequency=loss_weight)
    return loss, accuracy, prediction

def weighted_loss(self, logits, labels, number_class, frequency):
    """
    The reference paper is : https://arxiv.org/pdf/1411.4734.pdf 
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    we weight each pixels by alpha_c
    Inputs: 
    logits is the output from the inference, which is the output of the decoder layers without softmax.
    labels: true label information 
    number_class: In the CamVid data set, it's 11 classes, or 12, because class 11 seems to be background? 
    frequency: is the frequency of each class
    Outputs:
    Loss
    Accuracy
    """
    # raw_logits_reshape = tf.reshape(logits, [-1, number_class])
    # label_flatten = tf.reshape(labels, [-1])
    # #supposed 2 is the ignored label
    # indices=tf.squeeze(tf.where(tf.not_equal(label_flatten, -1)), 1)
    # label_flatten = tf.cast(tf.gather(label_flatten, indices), tf.int64)
    # logits_reshape = tf.gather(raw_logits_reshape, indices)
    # cross_entropy_mean = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_reshape,labels=label_flatten,name="entropy")))

    raw_logits_reshape = tf.reshape(logits, [-1, number_class])
    label_flatten = tf.reshape(labels, [-1])
    #supposed 2 is the ignored label
    indices=tf.squeeze(tf.where(tf.not_equal(label_flatten, -1)), 1)
    label_flatten = tf.cast(tf.gather(label_flatten, indices), tf.int64)
    logits_reshape = tf.gather(raw_logits_reshape, indices)
    label_onehot = tf.one_hot(label_flatten, depth=self.n_classes)
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot,
                                                             logits=logits_reshape,
                                                             pos_weight=frequency)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # tf.summary.scalar('loss', cross_entropy_mean)
    # correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    # accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    # tf.summary.scalar('accuracy', accuracy)

    # logits_reshape = tf.reshape(logits, [-1, number_class])
    # label_flatten = tf.reshape(labels, [-1])
    # label_onehot = tf.one_hot(label_flatten, depth=self.n_classes)
    # loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot,
    #                                                                logits=logits_reshape,
    #                                                                pos_weight=frequency))
    # correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    # accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

    # return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)
    return cross_entropy_mean, 0, 0


def labels_look_table(labels_path):
    # I think is better to save the scaler model, load this model here and then applied to all data
    labels = load_tiff_image(labels_path)
    mask_classes2_zero = (labels == 5)
    # print(np.unique(mask_classes2_zero))
    labels[mask_classes2_zero] = 0
    # mask_gan = np.load('mask_gans_trn.npy')
    # plt.figure('Maks ')
    # plt.imshow(mask_gan)
    # trn_labels = resampler(labels.copy(), 'uint8')[mask_gan == 1]
    # trn_labels = trn_labels[trn_labels != 0]
    # import collections
    # ctr = collections.Counter(trn_labels)
    # print(ctr)
    # plt.show(block=False)

    # create mapping of characters to integers (0-25) and the reverse
    print("unique labels for creating training samples --->", np.unique(labels))
    classes = np.unique(labels)
    classes = np.delete(classes, 0)
    print("unique labels for creating training samples --->", classes)
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    print (labels2new_labels)
    print (new_labels2labels)

    return labels2new_labels, new_labels2labels


def load_data4FCN_CV(self, samples_list, sample_index=None, labels2new_labels=None):
    from shutil import copyfile
    if sample_index is None:
        idxc = np.random.randint(0, len(samples_list))
    else:
        idxc = sample_index
    # src = samples_list[idxc]
    # print (src)
    # datasets_root = '/mnt/Data/Pix2Pix_datasets/Semi_Exp/' 
    # dataset_name = '05may2016_C01_synthesize_semisupervised_multitemporal/'
    # dst_root = datasets_root + dataset_name + 'Training/'
    data_Dic = np.load(samples_list[idxc]).item()
    labels = np.array(data_Dic['labels']).astype('uint8')
    img_A = np.array(data_Dic['img_A']).astype('float32')
    img_B = np.array(data_Dic['img_B']).astype('float32')
    img_SO = np.concatenate((img_A, img_B), axis=2)
    img_SO, labels = preprocess_S_and_O_and_L(img_SO, labels, load_size=self.load_size, fine_size=self.image_size)
    img_A = img_SO[:, :, :self.input_c_dim]
    img_B = img_SO[:, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

    if labels2new_labels is None:
        return img_A, img_B

    mask_classes2_zero = (labels == 5) + (labels == 10)
    labels[mask_classes2_zero] = 0
    num_labels = sum(labels.ravel() != 0)

    if (num_labels == 0):
        # generate dummy data
        # copyfile(src, dst_root + 'unsupervised/' + str(idxc) + '.npy')
        beta = 0.0
        dummy_labels = np.random.randint(-1, self.n_classes, size=labels.shape)
        return img_A, img_B, dummy_labels, beta
    # copyfile(src, dst_root + 'supervised/' + str(idxc) + '.npy')
    beta = 1.0
        
    new_labels = -1 * np.ones_like(labels)
    classes = np.unique(labels)
    for clas in classes:
        if clas != 0:
            new_labels[labels == clas] = labels2new_labels[clas]

    # print (new_labels)
    return img_A, img_B, new_labels, beta


def load_data4FCN_CV_cropping(self, samples_list, sample_index=None, labels2new_labels=None):
    from shutil import copyfile
    if sample_index is None:
        idxc = np.random.randint(0, len(samples_list))
    else:
        idxc = sample_index
    # src = samples_list[idxc]
    # print (src)
    # datasets_root = '/mnt/Data/Pix2Pix_datasets/Semi_Exp/' 
    # dataset_name = '05may2016_C01_synthesize_semisupervised_multitemporal/'
    # dst_root = datasets_root + dataset_name + 'Training/'
    data_Dic = np.load(samples_list[idxc]).item()
    labels = np.array(data_Dic['labels']).astype('uint8')
    img_A = np.array(data_Dic['img_A']).astype('float32')
    img_B = np.array(data_Dic['img_B']).astype('float32')
    img_SO = np.concatenate((img_A, img_B), axis=2)
    img_SO, labels = preprocess_S_and_O_and_L_cropping(img_SO, labels, load_size=self.load_size, fine_size=self.image_size)
    img_A = img_SO[:, :, :self.input_c_dim]
    img_B = img_SO[:, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

    if labels2new_labels is None:
        return img_A, img_B

    mask_classes2_zero = (labels == 5) + (labels == 10)
    labels[mask_classes2_zero] = 0
    num_labels = sum(labels.ravel() != 0)

    if (num_labels == 0):
        # generate dummy data
        # copyfile(src, dst_root + 'unsupervised/' + str(idxc) + '.npy')
        beta = 0.0
        dummy_labels = np.random.randint(-1, self.n_classes, size=labels.shape)
        return img_A, img_B, dummy_labels, beta
    # copyfile(src, dst_root + 'supervised/' + str(idxc) + '.npy')
    beta = 1.0
        
    new_labels = -1 * np.ones_like(labels)
    classes = np.unique(labels)
    for clas in classes:
        if clas != 0:
            new_labels[labels == clas] = labels2new_labels[clas]

    # print (new_labels)
    return img_A, img_B, new_labels, beta


def validate_FCN_CV(self, data_val_list, labels2new_labels):
        Loss = []
        Pred = []
        Ref = []
        for patch_dir in data_val_list:
            data_Dic = np.load(patch_dir).item()
            labels = np.array(data_Dic['labels'])
            mask_classes2_zero = (labels == 5) + (labels == 10)
            labels[mask_classes2_zero] = 0 # -1: means background

            if sum(labels.ravel() != 0) > 0:
                img_A = np.array(data_Dic['img_A']).astype('float32')
                img_B = np.array(data_Dic['img_B']).astype('float32')
                batch_images = np.concatenate((img_A.reshape(1, 256, 256, self.input_c_dim),
                                              img_B.reshape(1, 256, 256, self.output_c_dim)),
                                              axis=3)
                new_labels = -1 * np.ones_like(labels)
                classes = np.unique(labels)
                for clas in classes:
                    if clas != 0:
                        new_labels[labels == clas] = labels2new_labels[clas]

                loss, pred = self.sess.run([self.fcn_loss, self.FCN_logits_sample],
                                           feed_dict={self.real_data: batch_images, self.labels: new_labels, self.dropout: False})
                # print(np.shape(pred))
                new_labels = new_labels.ravel()
                pred = pred.reshape(-1, self.n_classes)
                pred = pred[new_labels != -1]
                Pred.append(np.argmax(pred, axis=1))                
                Ref.append(new_labels[new_labels != -1])
                Loss.append(loss)
                # print (np.unique(new_labels))

        Ref = np.concatenate(Ref).ravel()
        Pred = np.concatenate(Pred).ravel()
        print('metrics for fake data ... ')
        idx = range(len(Ref))
        np.random.shuffle(idx)
        compute_metrics(Ref, Pred)
        print(np.concatenate((Pred[idx].reshape(-1, 1), Ref[idx].reshape(-1, 1),), axis=1))

        return np.mean(Loss)


def validate_FCN_CV_batchsize2(self, data_val_list, labels2new_labels, real=False):
        Loss = []
        Pred = []
        Ref = []
        Mask = []
        w = 32
        d = (self.image_size - w)//2
        for patch_dir in data_val_list:
            data_Dic = np.load(patch_dir).item()
            mask = np.array(data_Dic['mask'])
            mask = mask[d:d+w, d:d+w]
            if np.sum(mask.ravel() == 2) == 0:
                continue

            labels = np.array(data_Dic['labels'])
            mask_classes2_zero = (labels == 5) + (labels == 10)
            labels[mask_classes2_zero] = 0 # -1: means background
            label_central = labels.copy()
            label_central = label_central[d:d+w, d:d+w]
            if sum(label_central.ravel() != 0) == 0:
                continue

            img_A = np.array(data_Dic['img_A']).astype('float32')
            img_B = np.array(data_Dic['img_B']).astype('float32')
            img_A = img_A.reshape(1, self.image_size, self.image_size, self.input_c_dim)
            img_B = img_B.reshape(1, self.image_size, self.image_size, self.output_c_dim)
            img_A = np.concatenate((img_A, img_A), axis=0)
            img_B = np.concatenate((img_B, img_B), axis=0)
            batch_images = np.concatenate((img_A, img_B), axis=3)
            new_labels = -1 * np.ones_like(labels)
            classes = np.unique(labels)
            for clas in classes:
                if clas != 0:
                    new_labels[labels == clas] = labels2new_labels[clas]

            new_labels = new_labels.reshape(1, self.image_size, self.image_size)
            new_labels = np.concatenate((new_labels, -1 * np.ones_like(new_labels)), axis=0)
            if real:
                loss, pred = self.sess.run([self.fcn_loss_real, self.FCN_logits_real],
                                           feed_dict={self.real_data: batch_images, self.labels: new_labels, self.dropout: False})
            else:
                loss, pred = self.sess.run([self.fcn_loss_fake, self.FCN_logits_fake],
                                           feed_dict={self.real_data: batch_images, self.labels: new_labels, self.dropout: True})
            # print(np.shape(pred))
            new_labels = new_labels[0][d:d+w, d:d+w].ravel()
            new_labels = new_labels[mask.ravel() == 2]
            pred = pred[0][d:d+w, d:d+w]
            # print (pred.shape)
            pred = pred.reshape(-1, self.n_classes)
            pred = pred[mask.ravel() == 2]
            pred = pred[new_labels != -1]
            Pred.append(np.argmax(pred, axis=1))                
            Ref.append(new_labels[new_labels != -1])
            Loss.append(loss)
            # print (np.unique(new_labels))

        Ref = np.concatenate(Ref).ravel()
        Pred = np.concatenate(Pred).ravel()
        print('metrics for fake data ... ')
        idx = range(len(Ref))
        np.random.shuffle(idx)
        compute_metrics(Ref, Pred)
        print(np.concatenate((Pred[idx].reshape(-1, 1), Ref[idx].reshape(-1, 1),), axis=1))

        return np.mean(Loss)


def validate_FCN_CV_batchsize(self, data_val_list, labels2new_labels, real=False):
        Loss = []
        Pred = []
        Ref = []
        Mask = []
        w = 64
        d = (self.image_size - w)//2
        for patch_dir in data_val_list:
            data_Dic = np.load(patch_dir).item()
            mask = np.array(data_Dic['mask'])
            mask = mask[d:d+w, d:d+w]
            if np.sum(mask.ravel() == 2) == 0:
                continue

            labels = np.array(data_Dic['labels'])
            mask_classes2_zero = (labels == 5) + (labels == 10)
            labels[mask_classes2_zero] = 0 # -1: means background
            label_central = labels.copy()
            label_central = label_central[d:d+w, d:d+w]
            if sum(label_central.ravel() != 0) == 0:
                continue

            img_A = np.array(data_Dic['img_A']).astype('float32')
            img_B = np.array(data_Dic['img_B']).astype('float32')
            img_A = img_A.reshape(1, 256, 256, self.input_c_dim)
            img_B = img_B.reshape(1, 256, 256, self.output_c_dim)
            img_A = np.concatenate((img_A, img_A), axis=0)
            img_B = np.concatenate((img_B, img_B), axis=0)
            batch_images = np.concatenate((img_A, img_B), axis=3)
            new_labels = -1 * np.ones_like(labels)
            classes = np.unique(labels)
            for clas in classes:
                if clas != 0:
                    new_labels[labels == clas] = labels2new_labels[clas]

            new_labels = new_labels.reshape(1, 256, 256)
            new_labels = np.concatenate((new_labels, -1 * np.ones_like(new_labels)), axis=0)
            if real:
                loss, pred = self.sess.run([self.fcn_loss_real, self.FCN_logits_real],
                                           feed_dict={self.real_data: batch_images, self.labels: new_labels, self.dropout: False})
            else:
                loss, pred = self.sess.run([self.fcn_loss_fake, self.FCN_logits_fake],
                                           feed_dict={self.real_data: batch_images, self.labels: new_labels, self.dropout_g: True})
            # print(np.shape(pred))
            new_labels = new_labels[0][d:d+w, d:d+w].ravel()
            new_labels = new_labels[mask.ravel() == 2]
            pred = pred[0][d:d+w, d:d+w]
            # print (pred.shape)
            pred = pred.reshape(-1, self.n_classes)
            pred = pred[mask.ravel() == 2]
            pred = pred[new_labels != -1]
            Pred.append(np.argmax(pred, axis=1))                
            Ref.append(new_labels[new_labels != -1])
            Loss.append(loss)
            # print (np.unique(new_labels))

        Ref = np.concatenate(Ref).ravel()
        Pred = np.concatenate(Pred).ravel()
        print('metrics for fake data ... ')
        idx = range(len(Ref))
        np.random.shuffle(idx)
        compute_metrics(Ref, Pred)
        print(np.concatenate((Pred[idx].reshape(-1, 1), Ref[idx].reshape(-1, 1),), axis=1))

        return np.mean(Loss)


def validate_FCN_CV_batchsize_discriminator(self, data_val_list, labels2new_labels, real=False):
        Loss = []
        Pred = []
        Ref = []
        Mask = []
        w = 64
        d = (self.image_size - w)//2
        for patch_dir in data_val_list:
            data_Dic = np.load(patch_dir).item()
            mask = np.array(data_Dic['mask'])
            mask = mask[d:d+w, d:d+w]
            if np.sum(mask.ravel() == 2) == 0:
                continue

            labels = np.array(data_Dic['labels'])
            mask_classes2_zero = (labels == 5) + (labels == 10)
            labels[mask_classes2_zero] = 0 # -1: means background
            label_central = labels.copy()
            label_central = label_central[d:d+w, d:d+w]
            if sum(label_central.ravel() != 0) == 0:
                continue

            img_A = np.array(data_Dic['img_A']).astype('float32')
            img_B = np.array(data_Dic['img_B']).astype('float32')
            img_A = img_A.reshape(1, 256, 256, self.input_c_dim)
            img_B = img_B.reshape(1, 256, 256, self.output_c_dim)
            img_A = np.concatenate((img_A, img_A), axis=0)
            img_B = np.concatenate((img_B, img_B), axis=0)
            batch_images = np.concatenate((img_A, img_B), axis=3)
            new_labels = -1 * np.ones_like(labels)
            classes = np.unique(labels)
            for clas in classes:
                if clas != 0:
                    new_labels[labels == clas] = labels2new_labels[clas]

            new_labels = new_labels.reshape(1, 256, 256)
            new_labels = np.concatenate((new_labels, -1 * np.ones_like(new_labels)), axis=0)
            if real:
                loss, pred = self.sess.run([self.fcn_loss_real, self.FCN_logits_real],
                                           feed_dict={self.real_data: batch_images, self.labels: new_labels, self.dropout: False})
            else:
                loss, pred = self.sess.run([self.fcn_loss_fake, self.D_logits_class_],
                                           feed_dict={self.real_data: batch_images, self.labels: new_labels, self.dropout_d: False, self.dropout_g: True})
            # print(np.shape(pred))
            new_labels = new_labels[0][d:d+w, d:d+w].ravel()
            new_labels = new_labels[mask.ravel() == 2]
            pred = pred[0][d:d+w, d:d+w]
            # print (pred.shape)
            pred = pred.reshape(-1, self.n_classes)
            pred = pred[mask.ravel() == 2]
            pred = pred[new_labels != -1]
            Pred.append(np.argmax(pred, axis=1))                
            Ref.append(new_labels[new_labels != -1])
            Loss.append(loss)
            # print (np.unique(new_labels))

        Ref = np.concatenate(Ref).ravel()
        Pred = np.concatenate(Pred).ravel()
        print('metrics for fake data ... ')
        idx = range(len(Ref))
        np.random.shuffle(idx)
        compute_metrics(Ref, Pred)
        print(np.concatenate((Pred[idx].reshape(-1, 1), Ref[idx].reshape(-1, 1),), axis=1))

        return np.mean(Loss)



def validate_generated_samplesCV(self, data_val_list, labels2new_labels, real=True):
        errC_real = []
        errC_fake = []
        pred_Real = []
        pred_Fake = []
        reference = []
        for patch_dir in data_val_list:
            data_Dic = np.load(patch_dir).item()
            labels_patch = np.array(data_Dic['labels']).ravel()
            labels = labels_patch.copy()
            # print (len(labels))
            eli_classes = (labels != 0) * (labels != 5) * (labels != 10)
            labels = labels[eli_classes]
            if len(labels) > 0:
                img_A = np.array(data_Dic['img_A']).astype('float32')
                img_B = np.array(data_Dic['img_B']).astype('float32')
                idx_samples = np.arange(len(labels_patch))
                batch_images = np.concatenate((img_A.reshape(1, 256, 256, self.input_c_dim),
                                              img_B.reshape(1, 256, 256, self.output_c_dim)),
                                              axis=3)
                idx_samples = idx_samples[eli_classes]
                # labels = labels[labels != 0]
                # idx_samples = idx_samples[labels != 5]
                # labels = labels[labels != 5]
                # idx_samples = idx_samples[labels != 10]
                # labels = labels[labels != 10]
                # min_classes = (labels==4) + (labels==9) + (labels==11)
                # labels[~ min_classes] = 0
                # print("unique labels for patch --->", np.unique(labels))
                new_labels = np.zeros_like(labels)
                idx_samples = idx_samples.reshape(-1, 1)
                classes = np.unique(labels)
                for clas in classes:
                    new_labels[labels == clas] = labels2new_labels[clas]

                if real:
                    pred_real, pred_fake, c_loss_fake, c_loss_real = self.sess.run([self.C_Real, self.C_Fake, self.c_loss_Fake, self.c_loss_Real],
                        feed_dict={self.real_data: batch_images, self.labels_classifier: new_labels, self.dropout: False, self.idx_samples: idx_samples, self.beta: 1})
                    pred_Real.append(np.argmax(pred_real, axis=1))
                else:
                    pred_fake, c_loss_fake, c_loss_real = self.sess.run([self.C_Fake, self.c_loss_Fake, self.c_loss_Real],
                        feed_dict={self.real_data: batch_images, self.labels_classifier: new_labels, self.dropout: False, self.idx_samples: idx_samples, self.beta: 1})
                pred_Fake.append(np.argmax(pred_fake, axis=1))
                reference.append(new_labels)
                errC_real.append(c_loss_real)
                errC_fake.append(c_loss_fake)
                # print (np.unique(new_labels))

        reference = np.concatenate(reference).ravel()
        pred_Fake = np.concatenate(pred_Fake).ravel()
        print('metrics for fake data ... ')
        compute_metrics(reference, pred_Fake)
        print(np.concatenate((pred_Fake.reshape(-1, 1), reference.reshape(-1, 1),), axis=1))
        if real:
            pred_Real = np.concatenate(pred_Real).ravel()
            print('metrics for real data ... ')
            # print(np.unique(pred_Fake))
            print(np.concatenate((pred_Real.reshape(-1, 1), reference.reshape(-1, 1)), axis=1))
            compute_metrics(reference, pred_Real)

        return np.mean(errC_real), np.mean(errC_fake)


def validate_generated_samplesCV_oneClass(self, data_val_list, labels2new_labels, real=True):
        errC_real = []
        errC_fake = []
        pred_Real = []
        pred_Fake = []
        reference = []
        for patch_dir in data_val_list:
            data_Dic = np.load(patch_dir).item()
            labels_patch = np.array(data_Dic['labels']).ravel()
            labels = labels_patch.copy()
            # print (len(labels))
            eli_classes = (labels != 0)
            labels = labels[eli_classes]
            if len(labels) > 0:
                img_A = np.array(data_Dic['img_A']).astype('float32')
                img_B = np.array(data_Dic['img_B']).astype('float32')
                idx_samples = np.arange(len(labels_patch))
                batch_images = np.concatenate((img_A.reshape(1, 256, 256, self.input_c_dim),
                                              img_B.reshape(1, 256, 256, self.output_c_dim)),
                                              axis=3)
                idx_samples = idx_samples[eli_classes]
                labels[labels != 2] = 0 # Set others label to zero
                new_labels = np.zeros_like(labels)
                idx_samples = idx_samples.reshape(-1, 1)
                classes = np.unique(labels)
                for clas in classes:
                    new_labels[labels == clas] = labels2new_labels[clas]

                if real:
                    pred_real, pred_fake, c_loss_fake, c_loss_real = self.sess.run([self.C_Real, self.C_Fake, self.c_loss_Fake, self.c_loss_Real],
                        feed_dict={self.real_data: batch_images, self.labels_classifier: new_labels, self.dropout: False, self.idx_samples: idx_samples, self.beta: 1})
                    pred_Real.append(np.argmax(pred_real, axis=1))
                else:
                    pred_fake, c_loss_fake, c_loss_real = self.sess.run([self.C_Fake, self.c_loss_Fake, self.c_loss_Real],
                        feed_dict={self.real_data: batch_images, self.labels_classifier: new_labels, self.dropout: False, self.idx_samples: idx_samples, self.beta: 1})
                pred_Fake.append(np.argmax(pred_fake, axis=1))
                reference.append(new_labels)
                errC_real.append(c_loss_real)
                errC_fake.append(c_loss_fake)
                # print (np.unique(new_labels))

        reference = np.concatenate(reference).ravel()
        pred_Fake = np.concatenate(pred_Fake).ravel()
        print('metrics for fake data ... ')
        compute_metrics(reference, pred_Fake)
        print(np.concatenate((pred_Fake.reshape(-1, 1), reference.reshape(-1, 1),), axis=1))
        if real:
            pred_Real = np.concatenate(pred_Real).ravel()
            print('metrics for real data ... ')
            # print(np.unique(pred_Fake))
            print(np.concatenate((pred_Real.reshape(-1, 1), reference.reshape(-1, 1)), axis=1))
            compute_metrics(reference, pred_Real)

        return np.mean(errC_real), np.mean(errC_fake)


def load_data4Validation(self, samples_list, labels2new_labels):
    val_Data = np.zeros((256**2 * len(samples_list), self.n_features))
    val_Labels = np.zeros((256**2 * len(samples_list)))
    index = np.arange(0, 256**2 * len(samples_list))
    for idxc in range(0, len(samples_list)):
        data_Dic = np.load(samples_list[idxc]).item()
        labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
        opt_patch = np.array(data_Dic['opt']).astype('float32').reshape(len(labels_patch), self.n_features)
        labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
        val_Data[idxc*256**2:(idxc+1)*256**2] = opt_patch
        val_Labels[idxc*256**2:(idxc+1)*256**2] = labels_patch
    # remove background
    index = index[val_Labels != 0]
    val_Data = val_Data[val_Labels != 0]
    val_Labels = val_Labels[val_Labels != 0]
    index = index[val_Labels != 5]
    val_Data = val_Data[val_Labels != 5]
    val_Labels = val_Labels[val_Labels != 5]
    index = index[val_Labels != 10]
    val_Data = val_Data[val_Labels != 10]
    val_Labels = val_Labels[val_Labels != 10]
    # val_Labels = np.array(val_Labels)
    # print (np.shape(val_Labels))
    # print (np.unique(val_Labels))
    # print (val_Labels[0])
    # _, counter = np.unique(val_Labels, return_counts=True)
    # counter=collections.Counter(val_Labels)
    # print(counter)
    # Balance data
    trn_samples, trn_labels, idx_samples = balance_data(val_Data, val_Labels, index=index, samples_per_class=10000)
    # # convert labels index and also one-hot encoding.
    trn_labels_one_hot = np.zeros((len(trn_labels), self.n_classes), dtype='float32')
    for idx_l in range(len(trn_labels)):
        one_hot = labels2new_labels[trn_labels[idx_l]]
        trn_labels_one_hot[idx_l, one_hot] = 1
    return trn_samples, trn_labels_one_hot


def create_training_samples4Classifier_Quemadas(image_path, labels_path, mask_path):
    # echarle un ojo despues
    img, _ = load_sentinel2(image_path)
    mask = load_tiff_image(mask_path)
    labels = load_tiff_image(labels_path)
    rows, cols, bands = img.shape
    img = img.reshape(rows * cols, bands)
    mask = mask.ravel()
    labels = labels.ravel()
    img = img[mask == 1]  # here index 1 means training samples
    labels = labels[mask == 1]
    img = img[labels != 5]
    labels = labels[labels != 5]
    img = img[labels != 10]
    labels = labels[labels != 10]
    # create mapping of characters to integers (0-25) and the reverse
    print (np.unique(labels))
    classes = np.unique(labels)
    num_classes = len(np.unique(labels))
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    print (labels2new_labels)
    print (new_labels2labels)
    np.save('labels2new_labels', labels2new_labels)
    np.save('new_labels2labels', new_labels2labels)
    # print ('dictionaries were saved')
    new_labels = np.zeros_like(labels)
    for i in classes:
        new_labels[labels == i] = labels2new_labels[i]

    scaler = pre.MinMaxScaler((-1, 1)).fit(img)
    img = np.float32(scaler.transform(img))

    index = range(len(new_labels))
    np.random.shuffle(index)
    new_labels = new_labels[index]
    img = img[index]
    n_samplesxtrn = int(0.6 * len(new_labels))
    n_samplesxval = len(new_labels) - n_samplesxtrn
    out_trn_data = np.zeros((n_samplesxtrn, img.shape[1]))
    out_trn_labels = np.zeros((n_samplesxtrn))
    out_val_data = np.zeros((n_samplesxval+2, img.shape[1]))
    out_val_labels = np.zeros((n_samplesxval+2))
    total_trn = 0
    total_val = 0
    count = 0
    for i in np.unique(new_labels):
        print (count)
        count += 1
        n_samples = np.sum(new_labels == i)
        n_samplesxtrn = int(0.6 * n_samples)
        n_samplesxval = n_samples - n_samplesxtrn
        samplesxclass = img[new_labels == i]
        labelsxclass = new_labels[new_labels == i]

        trn_samplesxclass = samplesxclass[:n_samplesxtrn]
        trn_labelsxclass = labelsxclass[:n_samplesxtrn]
        val_samplesxclass = samplesxclass[n_samplesxtrn:n_samples]
        val_labelsxclass = labelsxclass[n_samplesxtrn:n_samples]

        total_trn += n_samplesxtrn
        total_val += n_samplesxval

        out_trn_data[total_trn - n_samplesxtrn: total_trn] = trn_samplesxclass
        out_trn_labels[total_trn - n_samplesxtrn: total_trn] = trn_labelsxclass

        out_val_data[total_val - n_samplesxval: total_val] = val_samplesxclass
        out_val_labels[total_val - n_samplesxval: total_val] = val_labelsxclass

    # Balance the training data
    out_trn_data, out_trn_labels, _ = balance_data(out_trn_data,
                                                   out_trn_labels,
                                                   samples_per_class=10000)

    # convert class vectors to binary class matrices
    out_trn_labels = keras.utils.to_categorical(out_trn_labels, num_classes)
    out_val_labels = keras.utils.to_categorical(out_val_labels, num_classes)

    return out_trn_data, out_trn_labels, out_val_data, out_val_labels


def create_training_samples_Classifier_onehot(image_path, labels_path):
    # I think is better to save the scaler model, load this model here and then applied to all data
    scaler = joblib.load("opt_05may2016_scaler.pkl")
    img, _ = load_landsat(image_path)
    rows, cols, bands = img.shape
    img = img.reshape(rows * cols, bands)
    print(img.max(), img.min())
    img = np.float32(scaler.transform(img))
    # print(img)
    print(img.max(), img.min())
    mask = np.load('mask_gans_trn.npy')
    # plt.figure()
    # plt.imshow(mask)
    # plt.show(block=True)
    labels = load_tiff_image(labels_path)
    labels = resampler(labels, 'uint8')
    mask = mask.ravel()
    labels = labels.ravel()
    img = img[mask == 1]  # here index 1 means training samples
    labels = labels[mask == 1]
    img = img[labels != 0]
    labels = labels[labels != 0]
    img = img[labels != 5]
    labels = labels[labels != 5]
    img = img[labels != 10]
    labels = labels[labels != 10]
    # create mapping of characters to integers (0-25) and the reverse
    print (np.unique(labels))
    classes = np.unique(labels)
    num_classes = len(np.unique(labels))
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    print (labels2new_labels)
    print (new_labels2labels)
    np.save('labels2new_labels', labels2new_labels)
    np.save('new_labels2labels', new_labels2labels)
    # print ('dictionaries were saved')
    new_labels = np.zeros_like(labels)
    for i in classes:
        new_labels[labels == i] = labels2new_labels[i]

    index = range(len(new_labels))
    np.random.shuffle(index)
    new_labels = new_labels[index]
    img = img[index]
    n_samplesxtrn = int(0.6 * len(new_labels))
    n_samplesxval = len(new_labels) - n_samplesxtrn
    out_trn_data = np.zeros((n_samplesxtrn, img.shape[1]))
    out_trn_labels = np.zeros((n_samplesxtrn))
    out_val_data = np.zeros((n_samplesxval+4, img.shape[1]))
    out_val_labels = np.zeros((n_samplesxval+4))
    total_trn = 0
    total_val = 0
    count = 0
    for i in np.unique(new_labels):
        print (count)
        count += 1
        n_samples = np.sum(new_labels == i)
        n_samplesxtrn = int(0.6 * n_samples)
        n_samplesxval = n_samples - n_samplesxtrn
        samplesxclass = img[new_labels == i]
        labelsxclass = new_labels[new_labels == i]

        trn_samplesxclass = samplesxclass[:n_samplesxtrn]
        trn_labelsxclass = labelsxclass[:n_samplesxtrn]
        val_samplesxclass = samplesxclass[n_samplesxtrn:n_samples]
        val_labelsxclass = labelsxclass[n_samplesxtrn:n_samples]

        total_trn += n_samplesxtrn
        total_val += n_samplesxval

        out_trn_data[total_trn - n_samplesxtrn: total_trn] = trn_samplesxclass
        out_trn_labels[total_trn - n_samplesxtrn: total_trn] = trn_labelsxclass

        out_val_data[total_val - n_samplesxval: total_val] = val_samplesxclass
        out_val_labels[total_val - n_samplesxval: total_val] = val_labelsxclass

    # Balance the training data
    out_trn_data, out_trn_labels, _ = balance_data(out_trn_data,
                                                   out_trn_labels,
                                                   samples_per_class=10000)

    # convert class vectors to binary class matrices
    out_trn_labels = keras.utils.to_categorical(out_trn_labels, num_classes)
    out_val_labels = keras.utils.to_categorical(out_val_labels, num_classes)

    return out_trn_data, out_trn_labels, out_val_data, out_val_labels, labels2new_labels, new_labels2labels

    # convert class vectors to binary class matrices
    # one_hot_labels = keras.utils.to_categorical(new_labels, num_classes)
    # np.save('trn_samples_classifier', img)
    # np.save('trn_labels_classifier', one_hot_labels)
    # min_classes = (labels==4) + (labels==9) + (labels==11)
    # labels[labels != min_classes] = 0
    # img = img[labels != 5]
    # labels = labels[labels != 5]
    # img = img[labels != 10]
    # labels = labels[labels != 10]

    # mask = mask.ravel()
    # labels = labels.ravel()
    # img = img[mask == 1]  # here index 1 means training samples
    # labels = labels[mask == 1]
    # img = img[labels != 0]
    # labels = labels[labels != 0]
    # img = img[labels != 5]
    # labels = labels[labels != 5]
    # img = img[labels != 10]
    # labels = labels[labels != 10]

def create_training_samples_Classifier(image_path, labels_path):
    # I think is better to save the scaler model, load this model here and then applied to all data
    scaler = joblib.load("opt_may2016_scaler.pkl")
    img, _ = load_landsat(image_path)
    rows, cols, bands = img.shape
    img = img.reshape(rows * cols, bands)
    print(img.max(), img.min())
    img = np.float32(scaler.transform(img))
    # print(img)
    print(img.max(), img.min())
    mask = np.load('mask_gans_trn.npy')
    # plt.figure()
    # plt.imshow(mask)
    # plt.show(block=True)
    labels = load_tiff_image(labels_path)
    labels = resampler(labels, 'uint8')
    mask = mask.ravel()
    labels = labels.ravel()
    trn_data = img[mask == 1]  # here index 1 means training samples
    trn_labels = labels[mask == 1]
    eli_classes = (trn_labels != 0) * (trn_labels != 5) * (trn_labels != 10)
    trn_labels = trn_labels[eli_classes]
    trn_data = trn_data[eli_classes]
    # create mapping of characters to integers (0-25) and the reverse
    print("unique labels for creating training samples --->", np.unique(trn_labels))
    classes = np.unique(trn_labels)
    # num_classes = len(np.unique(trn_labels))
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    print (labels2new_labels)
    print (new_labels2labels)
    # np.save('labels2new_labels_4_9_11', labels2new_labels)
    # np.save('new_labels2labels_4_9_11', new_labels2labels)
    # print ('dictionaries were saved')
    new_labels = np.zeros_like(trn_labels)
    for i in classes:
        new_labels[trn_labels == i] = labels2new_labels[i]

    # Balance the training data
    out_trn_data, out_trn_labels, _ = balance_data(trn_data,
                                                   new_labels,
                                                   samples_per_class=10000)

    return out_trn_data, np.int32(out_trn_labels), labels2new_labels


def load_train_test_samples(image_path, labels_path):
    # I think is better to save the scaler model, load this model here and then applied to all data
    scaler = joblib.load("opt_may2016_scaler.pkl")
    img, _ = load_landsat(image_path)
    rows, cols, bands = img.shape
    img = img.reshape(rows * cols, bands)
    print(img.max(), img.min())
    img = np.float32(scaler.transform(img))
    # print(img)
    print(img.max(), img.min())
    mask = np.load('mask_gans_trn.npy')
    # plt.figure()
    # plt.imshow(mask)
    # plt.show(block=True)
    labels = load_tiff_image(labels_path)
    labels = resampler(labels, 'uint8')
    mask = mask.ravel()
    labels = labels.ravel()
    eli_classes = (labels != 0) * (labels != 5) * (labels != 10)
    labels = labels[eli_classes]
    data = img[eli_classes]
    mask = mask[eli_classes]
    trn_data = data[mask == 1]  # here index 1 means training samples
    trn_labels = labels[mask == 1]
    tst_data = data[mask == 2]  # here index 1 means training samples
    tst_labels = labels[mask == 2]
    # create mapping of characters to integers (0-25) and the reverse
    print("unique labels for creating training samples --->", np.unique(trn_labels))
    classes = np.unique(trn_labels)
    num_classes = len(np.unique(trn_labels))
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    print (labels2new_labels)
    print (new_labels2labels)
    # np.save('labels2new_labels_4_9_11', labels2new_labels)
    # np.save('new_labels2labels_4_9_11', new_labels2labels)
    # print ('dictionaries were saved')
    new_labels_trn = np.zeros_like(trn_labels)
    new_labels_tst = np.zeros_like(tst_labels)
    for i in classes:
        new_labels_trn[trn_labels == i] = labels2new_labels[i]
        new_labels_tst[tst_labels == i] = labels2new_labels[i]

    print(np.unique(new_labels_trn))
    print(np.unique(new_labels_tst))
    # Balance the training data
    # out_trn_data, out_trn_labels, _ = balance_data(trn_data,
                                                   # new_labels,
                                                   # samples_per_class=10000)

    return trn_data, np.int32(new_labels_trn), tst_data, np.int32(new_labels_tst)


def create_training_samples_Classifier_oneClass(image_path, labels_path):
    # I think is better to save the scaler model, load this model here and then applied to all data
    # label 3
    scaler = joblib.load("opt_may2016_scaler.pkl")
    img, _ = load_landsat(image_path)
    rows, cols, bands = img.shape
    img = img.reshape(rows * cols, bands)
    print(img.max(), img.min())
    img = np.float32(scaler.transform(img))
    # print(img)
    print(img.max(), img.min())
    mask = np.load('mask_gans_trn.npy')
    # plt.figure()
    # plt.imshow(mask)
    # plt.show(block=True)
    labels = load_tiff_image(labels_path)
    labels = resampler(labels, 'uint8')
    mask = mask.ravel()
    labels = labels.ravel()
    trn_data = img[mask == 1]  # here index 1 means training samples
    trn_labels = labels[mask == 1]
    trn_data = trn_data[trn_labels != 0]
    trn_labels = trn_labels[trn_labels != 0]
    trn_labels[trn_labels != 2] = 0 # Set others label to zero
    # create mapping of characters to integers (0-25) and the reverse
    print("unique labels for creating training samples --->", np.unique(trn_labels))
    classes = np.unique(trn_labels)
    # num_classes = len(np.unique(trn_labels))
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    print (labels2new_labels)
    print (new_labels2labels)
    # np.save('labels2new_labels_4_9_11', labels2new_labels)
    # np.save('new_labels2labels_4_9_11', new_labels2labels)
    # print ('dictionaries were saved')
    new_labels = np.zeros_like(trn_labels)
    num_samplesxclass = []
    for i in classes:
        num_samplesxclass.append(sum(trn_labels == i))
        new_labels[trn_labels == i] = labels2new_labels[i]

    samples_per_class = np.min(num_samplesxclass)
    print("num_samplesxclass --->", samples_per_class)
    # Balance the training data
    out_trn_data, out_trn_labels, _ = balance_data(trn_data,
                                                   new_labels,
                                                   samples_per_class=samples_per_class)

    return out_trn_data, np.int32(out_trn_labels), labels2new_labels


def evaluate_classifier_unsupervised(self, data_val_list, labels2new_labels, real=True):
    errC_real = []
    errC_fake = []
    pred_Real = []
    pred_Fake = []
    reference = []
    for patch_dir in data_val_list:
        data_Dic = np.load(patch_dir).item()
        labels = np.array(data_Dic['labels']).ravel()
        sar_patch = np.array(data_Dic['sar']).astype('float32')
        opt_patch = np.array(data_Dic['opt']).astype('float32')
        batch_images = np.concatenate((sar_patch.reshape(1, self.image_size,
                                                     self.image_size,
                                                     self.input_c_dim),
                                   opt_patch.reshape(1, self.image_size,
                                                     self.image_size,
                                                     self.output_c_dim)),
                                  axis=3)
        opt_patch_fake = self.sess.run([self.fake_B],
                                       feed_dict={self.real_data: batch_images})

        index = np.arange(len(labels))
        index = index[labels != 0]
        labels = labels[labels != 0]
        index = index[labels != 5]
        labels = labels[labels != 5]
        index = index[labels != 10]
        labels = labels[labels != 10]
        new_labels = np.zeros_like(labels)
        labels_one_hot = np.zeros((len(labels), self.n_classes), dtype='float32')
        for idx_l in range(len(labels)):
            one_hot = labels2new_labels[labels[idx_l]]
            new_labels[idx_l] = one_hot
            labels_one_hot[idx_l, one_hot] = 1

        # print(np.shape(opt_patch_fake))
        fake_data = np.reshape(opt_patch_fake, (-1, self.n_features))
        fake_data = fake_data[index]
        pred_fake, c_loss_fake = self.sess.run([self.C, self.c_loss],
                                                feed_dict={self.data_classifier: fake_data, self.labels_classifier: labels_one_hot, self.dropout: False})
        pred_Fake.append(np.argmax(pred_fake, axis=1))
        reference.append(new_labels)
        errC_fake.append(c_loss_fake)
        print (np.unique(new_labels))

        if (real):
            real_data = opt_patch.reshape(-1, self.n_features)
            real_data = real_data[index]
            pred_real, c_loss_real = self.sess.run([self.C, self.c_loss],
                                                   feed_dict={self.data_classifier: real_data, self.labels_classifier: labels_one_hot, self.dropout: False})
            pred_Real.append(np.argmax(pred_real, axis=1))
            errC_real.append(c_loss_real)

    reference = np.concatenate(reference).ravel()
    pred_Fake = np.concatenate(pred_Fake).ravel()
    print('metrics for fake data ... ')
    compute_metrics(reference, pred_Fake)
    print(np.concatenate((pred_Fake.reshape(-1, 1), reference.reshape(-1, 1),), axis=1))
    if real:
        pred_Real = np.concatenate(pred_Real).ravel()
        print('metrics for real data ... ')
        print(np.concatenate((pred_Real.reshape(-1, 1), reference.reshape(-1, 1)), axis=1))
        compute_metrics(reference, pred_Real)

    return np.mean(errC_real), np.mean(errC_fake)


def extract_patches(ksize,
                    mask,
                    cloud_mask,
                    output_folder,
                    sar,
                    opt,
                    num_patches=400,
                    show=False):
    rows, cols = mask.shape
    r, _ = np.where(mask == 1)
    top_row = r[0]
    mask_copy = mask.copy()
    pixel_index_img = np.arange(0, rows*cols, dtype='uint32').reshape(rows, cols)
    pixel_idx = pixel_index_img[mask != 0]  # seleciona pixels de interes
    mask = mask[mask != 0]
    row, col = np.uint32(pixel_idx / cols), np.uint32(pixel_idx % cols)
    tr = row-top_row-ksize/2 > 0
    row = row[tr]
    col = col[tr]
    mask = mask[tr]
    br = row+ksize/2+1 < rows
    row = row[br]
    col = col[br]
    mask = mask[br]
    lr = col-ksize/2 > 0
    row = row[lr]
    col = col[lr]
    mask = mask[lr]
    rr = col + ksize + 1 < cols
    row = row[rr]
    col = col[rr]
    mask = mask[rr]
    idx = np.arange(len(row))
    np.random.shuffle(idx)
#    print idx
    cont_patches = 0

    if show:
        plt.close('all')
#        plt.figure()
#        plt.imshow(mask_copy)
#        plt.show(block=False)
#        plt.pause(0.001)
#        plt.figure()
    m2 = mask_copy.copy()
    for i in idx:
        print cont_patches
        if cont_patches >= num_patches:
            break
        check_clouds = cloud_mask[row[i]-int(ksize/2):row[i]+int(ksize/2), col[i]:col[i]+int(ksize)]
        check_mask_copy = mask_copy[row[i]-int(ksize/2):row[i]+int(ksize/2), col[i]:col[i]+int(ksize)]

        if np.sum(check_clouds == 1) <= 10 and np.sum(check_mask_copy == 0) <= 1:
#            save_imgs_pars(output_folder, sar, opt, ksize, row[i], col[i], cont_patches)
            cont_patches += 1
#            m = cloud_mask.copy()
#            m2 = mask_copy.copy()
            m2[row[i]-int(ksize/2):row[i]+int(ksize/2), col[i]:col[i]+int(ksize)] = np.random.randint(255)
    if show:
        fig, ax = plt.subplots()
        ax.set_rasterized(True)
        plt.imshow(m2)
        plt.show()
        fig.tight_layout()
        plt.savefig('/home/jose/Drive/PUC/WorkPlace/ISSPRS/gans_trn_area.eps', dpi=100)
#        plt.pause(0.00000001)
#            plt.close()
    print cont_patches


#    row_trn = row[idx]
#    col_trn = col[idx]
#
#
#    if show:
#        mask_copy[row_trn, col_trn] = 5
#        plt.figure()
#        plt.imshow(m)
#        plt.show()

    return 0

# def extract_patches_stride(ksize,
#                            mask,
#                            cloud_mask,
#                            output_folder,
#                            sar,
#                            opt,
#                            show=False):
#     rows, cols = mask.shape
#     cont = 0
#     for row in range(0, rows-ksize-1, 23):
#         for col in range(0, cols-ksize-1, 23):
# #            print ("Patch  ...", np.sum(mask[row:row+ksize, col:col+ksize]))
#             if (np.sum(mask[row:row+ksize, col:col+ksize]) == 256**2) and (np.sum(cloud_mask[row:row+ksize, col:col+ksize])==0):
#                 print ("Patch valido ...", cont)
#                 patch_sar = sar[row:row+ksize, col:col+ksize, :]
#                 patch_opt = opt[row:row+ksize, col:col+ksize, :]
#                 patch_par = np.concatenate((patch_sar, patch_opt), axis=2)
#                 np.save(output_folder + str(cont), patch_par)
#                 cont += 1
#     return 0




    rows, cols = mask.shape
    cont = 0
    m2 = mask.copy()
    for row in range(0, rows - ksize - 1, 75):
        for col in range(0, cols - ksize - 1, 75):
            auxmask = mask[row:row + ksize, col:col + ksize]
            if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) > (ksize**2) / 1.1) and (np.sum(auxmask==0) < (ksize**2) / 1.01) and (np.sum(auxmask==2) == 0):
                print ("Patch valido ...", cont)
                patch_sar_t0 = sar_t0[row:row + ksize, col:col + ksize, :]
                patch_sar_t1 = sar_t1[row:row + ksize, col:col + ksize, :]
                patch_opt_t0 = opt_t0[row:row + ksize, col:col + ksize, :]
                patch_opt_t1 = opt_t1[row:row + ksize, col:col + ksize, :]
                patch_par = np.concatenate((patch_sar_t0, patch_sar_t1, patch_opt_t1, patch_opt_t0), axis=2)
                # patch_B = patch_opt_t1
                # patch_par = [patch_A, patch_B]
                # patch_par = [patch_sar_t0, patch_sar_t1, patch_opt_t0, patch_opt_t1]
                np.save(output_folder + str(cont), patch_par)
                cont += 1
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
    m2[mask==2] = 0
    plt.imshow(m2)
    plt.show()
    train_area = m2[m2!=0]
    print(train_area.shape)
    print ()
    return 0


# def extract_patches_4_classifier(
#         ksize,
#         mask,
#         mask_sar,
#         labels,
#         output_folder,
#         sar,
#         opt,
#         stride=128,
#         show=False):

#     print (output_folder)
#     rows, cols = mask.shape
#     cont = 0
#     m2 = mask.copy()
#     for row in range(0, rows - ksize - 1, stride):
#         for col in range(0, cols - ksize - 1, stride):
#             auxmask = mask[row:row + ksize, col:col + ksize]
#             if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) > (ksize**2) / 1.1) and (np.sum(auxmask==0) < (ksize**2) / 1.2) and (np.sum(auxmask==2) == 0):
#                 print ("Patch valido ...", cont)
#                 patch_tuple = {}
#                 patch_sar = sar[row:row + ksize, col:col + ksize, :]
#                 patch_tuple["sar"] = patch_sar
#                 patch_opt = opt[row:row + ksize, col:col + ksize, :]
#                 patch_tuple["opt"] = patch_opt
#                 patch_labels = labels[row:row + ksize, col:col + ksize]
#                 patch_tuple["labels"] = patch_labels
#                 # patch_par = [patch_sar, patch_opt, patch_labels]
#                 # patch_par = np.concatenate((patch_sar, patch_opt), axis=2)
#                 # np.save(output_folder + str(cont), patch_tuple)
#                 # Load read_dictionary = np.load('my_file.npy').item()
#                 cont += 1
#                 m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
#     # m2[mask==2] = 0
#     plt.figure()
#     plt.imshow(m2)
#     plt.show()
#     # train_area = m2[m2!=0]
#     # print(train_area.shape)
#     # print ()
#     return 0

def extract_patches_4_classifier(
        ksize,
        mask,
        mask_sar,
        labels,
        output_folder,
        sar,
        opt,
        stride=128,
        show=False):
    output_folder = output_folder + '/Training/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print (output_folder)
    rows, cols = mask.shape
    cont = 0
    m2 = np.zeros_like(mask)
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            auxmasksar = mask_sar[row:row + ksize, col:col + ksize]
            # if (np.sum(auxmasksar) > (ksize**2) / 1.1) and (np.sum(auxmask==1) > 100) and (np.sum(auxmask==2) == 0):
            if (np.sum(auxmask) >= (ksize**2 - 50)):
                print ("Patch valido ...", cont)
                patch_tuple = {}
                patch_sar = sar[row:row + ksize, col:col + ksize, :]
                patch_tuple["sar"] = patch_sar
                patch_opt = opt[row:row + ksize, col:col + ksize, :]
                patch_tuple["opt"] = patch_opt
                patch_labels = labels[row:row + ksize, col:col + ksize]
                patch_tuple["labels"] = patch_labels
                # patch_par = [patch_sar, patch_opt, patch_labels]
                # patch_par = np.concatenate((patch_sar, patch_opt), axis=2)
                np.save(output_folder + str(cont), patch_tuple)
                # Load read_dictionary = np.load('my_file.npy').item()
                cont += 1
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
    # m2[mask==2] = 0
    plt.figure()
    plt.imshow(m2)
    plt.show(block=False)
    # train_area = m2[m2!=0]
    # print(train_area.shape)
    # print ()
    return 0


def extract_patches_4_classifier_multitemporal(ksize,
                                               mask,
                                               mask_sar,
                                               labels,
                                               output_folder,
                                               img_A,
                                               img_B,
                                               stride=128,
                                               block=False):
    output_folder = output_folder + 'Training/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print (output_folder)
    rows, cols = mask.shape
    cont = 0
    m2 = np.zeros_like(mask)
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            auxmasksar = mask_sar[row:row + ksize, col:col + ksize]
            # if (np.sum(auxmasksar) > ksize**2 - 50) and (np.sum(auxmask==2) == 0):
            # if (np.sum(auxmask) >= (ksize**2 - 50)):
            if (np.sum(auxmasksar) > ksize**2 - 50) and (np.sum(auxmask==0) < ksize**2 // 1.1) and (np.sum(auxmask==2) == 0):
                print ("Patch valido ...", cont)
                patch_tuple = {}
                patch_tuple["img_A"] = img_A[row:row + ksize, col:col + ksize, :]
                patch_tuple["img_B"] = img_B[row:row + ksize, col:col + ksize, :]
                patch_tuple["labels"] = labels[row:row + ksize, col:col + ksize]
                np.save(output_folder + str(cont), patch_tuple)
                cont += 1
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
    # m2[mask==2] = 0
    plt.figure()
    plt.imshow(m2)
    plt.show(block=block)
    # train_area = m2[m2!=0]
    # print(train_area.shape)
    # print ()
    return 0

# if flip and trans_creterion > 0.90:
#         img_AB = np.rot90(img_AB, 3, (0, 1))
#         labels = np.rot90(labels, 3, (0, 1))
#     elif flip and trans_creterion > 0.85:
#         img_AB = np.rot90(img_AB, 2, (0, 1))
#         labels = np.rot90(labels, 2, (0, 1))
#     elif flip and trans_creterion > 0.80:
#         img_AB = np.rot90(img_AB, 1, (0, 1))
#         labels = np.rot90(labels, 1, (0, 1))
#     elif flip and trans_creterion > 0.75:
#         img_AB = np.flipud(img_AB)
#         labels = np.flipud(labels)
#     elif flip and trans_creterion > 0.50:
#         img_AB = np.fliplr(img_AB)
#         labels = np.fliplr(labels)
def data_augmentation(input_path=None, output_path=None, tranformations=None):
    # input_path = '/mnt/Data/Pix2Pix_datasets/Campo_Verde/Training/'
    # output_path = '/mnt/Data/Pix2Pix_datasets/Campo_Verde/Training_flip/'
    input_path = '/mnt/Data/Pix2Pix_datasets/Quemadas/Training/'
    output_path = '/mnt/Data/Pix2Pix_datasets/Quemadas/Training_flip/'
    input_patches = glob.glob(input_path + '*.npy')
    output_folder = output_path
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print (output_folder)
    
    cont = 0
    for patch in input_patches:
        # Read patches
        data_Dic = np.load(patch).item()
        sar_t0_Ori = np.array(data_Dic['sar_t0']).astype('float32')
        opt_t0_Ori = np.array(data_Dic['opt_t0']).astype('float32')
        sar_t1_Ori = np.array(data_Dic['sar_t1']).astype('float32')
        opt_t1_Ori = np.array(data_Dic['opt_t1']).astype('float32')
        labels_Ori = np.array(data_Dic['labels']).astype('uint8')
        # Apply transformation
        # Horizontal transformation
        patch_tuple = {}
        patch_tuple["sar_t0"] = np.flipud(sar_t0_Ori.copy())
        patch_tuple["opt_t0"] = np.flipud(opt_t0_Ori.copy())
        patch_tuple["sar_t1"] = np.flipud(sar_t1_Ori.copy())
        patch_tuple["opt_t1"] = np.flipud(opt_t1_Ori.copy())
        patch_tuple["labels"] = np.flipud(labels_Ori.copy())
        np.save(output_folder + 'h_' + str(cont), patch_tuple)
        # Vertical transformation
        patch_tuple = {}
        patch_tuple["sar_t0"] = np.fliplr(sar_t0_Ori.copy())
        patch_tuple["opt_t0"] = np.fliplr(opt_t0_Ori.copy())
        patch_tuple["sar_t1"] = np.fliplr(sar_t1_Ori.copy())
        patch_tuple["opt_t1"] = np.fliplr(opt_t1_Ori.copy())
        patch_tuple["labels"] = np.fliplr(labels_Ori.copy())
        np.save(output_folder + 'v_' + str(cont), patch_tuple)
        cont += 1

    return 0


def extract_patches_multitemporal_multiresolution(ksize,
                                                  mask,
                                                  mask_sar,
                                                  labels,
                                                  output_folder,
                                                  sar_t0,
                                                  opt_t0,
                                                  sar_t1,
                                                  opt_t1,
                                                  stride=128,
                                                  block=False):
    output_folder = output_folder + 'Training/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print (output_folder)
    rows, cols = mask.shape
    cont = 0
    m2 = np.zeros_like(mask)
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            auxmasksar = mask_sar[row:row + ksize, col:col + ksize]
            # if (np.sum(auxmasksar) > ksize**2 - 50) and (np.sum(auxmask==2) == 0):
            # if (np.sum(auxmask) >= (ksize**2 - 50)):
            if (np.sum(auxmasksar) > ksize**2 - 50) and (np.sum(auxmask==0) < ksize**2 // 1.1) and (np.sum(auxmask==2) == 0): # extract paches labeled
            # if (np.sum(auxmasksar) > ksize**2 - 50) and (np.sum(auxmask==0) < ksize**2 // 1.01): # extract paches labeled
                print ("Patch valido ...", cont)
                patch_tuple = {}
                patch_tuple["sar_t0"] = sar_t0[3*row:3*row + 3*ksize, 3*col:3*col + 3*ksize, :]
                patch_tuple["opt_t0"] = opt_t0[row:row + ksize, col:col + ksize, :]
                patch_tuple["sar_t1"] = sar_t1[3*row:3*row + 3*ksize, 3*col:3*col + 3*ksize, :]
                patch_tuple["opt_t1"] = opt_t1[row:row + ksize, col:col + ksize, :]
                patch_tuple["labels"] = labels[row:row + ksize, col:col + ksize]
                # np.save(output_folder + str(cont), patch_tuple)
                cont += 1
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
    # m2[mask==2] = 0
    plt.figure()
    plt.imshow(m2)
    plt.show(block=block)
    # train_area = m2[m2!=0]
    # print(train_area.shape)
    # print ()
    return 0


def extract_patches_4_testing_multitemporal(ksize,
                                            mask,
                                            mask_sar,
                                            labels,
                                            output_folder,
                                            sar_t0,
                                            opt_t0,
                                            sar_t1,
                                            opt_t1,
                                            stride=128,
                                            block=False):

    print (output_folder)
    rows, cols = mask.shape
    cont = 0
    m2 = np.zeros_like(mask)
    output_folder = output_folder + 'Testing/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            # if (np.sum(mask_sar[row:row + ksize, col:col + ksize])) == ksize**2 and (np.sum(auxmask==2) == ksize**2):
            if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) >= ksize**2 - 50) and (np.sum(auxmask == 2) >= 0.05 * ksize**2) and (np.sum(auxmask == 1) <= ksize**2 // 8):
                print ("Patch valido ...", cont)
                patch_tuple = {}
                patch_tuple["sar_t0"] = sar_t0[3*row:3*row + 3*ksize, 3*col:3*col + 3*ksize, :]
                patch_tuple["opt_t0"] = opt_t0[row:row + ksize, col:col + ksize, :]
                patch_tuple["sar_t1"] = sar_t1[3*row:3*row + 3*ksize, 3*col:3*col + 3*ksize, :]
                patch_tuple["opt_t1"] = opt_t1[row:row + ksize, col:col + ksize, :]
                patch_tuple["labels"] = labels[row:row + ksize, col:col + ksize]
                # np.save(output_folder + str(cont), patch_tuple)
                cont += 1
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
                # plt.figure("patches SAR")
                # plt.imshow(patch_tuple["sar_t0"][:, :, 0])
                # plt.show(block=False)
                # plt.figure("patches Opt")
                # plt.imshow(patch_tuple["opt_t0"][:, :, 3])
                # plt.show(block=True)

    print (m2.shape)
    plt.figure("patches Testing")
    plt.imshow(m2)
    plt.show(block=True)
    return 0

# def extract_patches_4_classifierQuemadas(
#         ksize,
#         mask,
#         mask_sar,
#         labels,
#         output_folder,
#         sar,
#         opt,
#         stride=128,
#         show=False):

#     print (output_folder)
#     rows, cols = mask.shape
#     cont = 0
#     m2 = mask.copy()
#     for row in range(0, rows - ksize - 1, stride):
#         for col in range(0, cols - ksize - 1, stride):
#             auxmask = mask[row:row + ksize, col:col + ksize]
#             if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) > (ksize**2) / 1.1) and (np.sum(auxmask==2) == 0) and (np.sum(auxmask==1) > 50):
#                 print ("Patch valido ...", cont)
#                 # patch_tuple = {}
#                 patch_sar = sar[row:row + ksize, col:col + ksize, :]
#                 # patch_tuple["sar"] = patch_sar
#                 patch_opt = opt[row:row + ksize, col:col + ksize, :]
#                 # patch_tuple["opt"] = patch_opt
#                 # patch_labels = labels[row:row + ksize, col:col + ksize]
#                 # patch_tuple["labels"] = patch_labels
#                 patch_tuple = np.concatenate((patch_sar, patch_opt), axis=2)
#                 np.save(output_folder + str(cont), patch_tuple)
#                 # Load read_dictionary = np.load('my_file.npy').item()
#                 cont += 1
#                 m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
#     # m2[mask==2] = 0
#     # figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi
#     # my_dpi = 300
#     print (m2.shape)
#     # plt.figure(figsize=(m2.shape[1]/my_dpi, m2.shape[0]/my_dpi), dpi=my_dpi)
#     plt.imshow(m2)
#     # plt.gca().set_axis_off()
#     # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     # plt.margins(0, 0)
#     # plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     # plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     # plt.savefig('quemadas_samples.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
#     plt.show(block=True)
#     # train_area = m2[m2!=0]
#     # print(train_area.shape)
#     # print ()
#     return 0


def extract_patches_4_testing(
        ksize,
        mask,
        mask_sar,
        labels,
        output_folder,
        sar,
        opt,
        stride=256,
        show=False):

    print (output_folder)
    rows, cols = mask.shape
    cont = 0
    m2 = np.zeros_like(mask)
    tenting_path = output_folder + 'Testing/'
    if not os.path.exists(tenting_path):
        os.makedirs(tenting_path)
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            # if (np.sum(mask_sar[row:row + ksize, col:col + ksize])) == ksize**2 and (np.sum(auxmask==2) == ksize**2):
            if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) == ksize**2) and (np.sum(auxmask == 2) >= 0.1 * ksize**2 and (np.sum(auxmask == 1) == 0)):
                print ("Patch valido ...", cont)
                patch_tuple = {}
                patch_sar = sar[row:row + ksize, col:col + ksize, :]
                patch_tuple["sar"] = patch_sar
                patch_opt = opt[row:row + ksize, col:col + ksize, :]
                patch_tuple["opt"] = patch_opt
                patch_labels = labels[row:row + ksize, col:col + ksize]
                patch_tuple["labels"] = patch_labels
                # patch_tuple = np.concatenate((patch_sar, patch_opt), axis=2)
                # Load read_dictionary = np.load('my_file.npy').item()
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
                print ("Training sample -->", cont)                
                np.save(tenting_path + str(cont), patch_tuple)
                cont += 1

    print (m2.shape)
    plt.figure()
    plt.imshow(m2)
    plt.show(block=True)
    return 0


def extract_patches_multitemporal_Quemandas(ksize,
                                            mask,
                                            mask_sar,
                                            output_folder,
                                            sar_t0,
                                            sar_t1,
                                            opt_t0,
                                            opt_t1,
                                            labels,
                                            stride=128,
                                            show=False):
    output_folder = output_folder + 'Training/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print (output_folder)
    rows, cols = mask.shape
    cont = 0
    m2 = mask.copy()
    tst_index = []
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) > (ksize**2) / 1.1) and (np.sum(auxmask==2) == 0) and (np.sum(auxmask==1) > 50):
                print ("Patch valido ...", cont)
                print ("Patch valido ...", cont)
                patch_tuple = {}
                patch_tuple["sar_t0"] = sar_t0[row:row + ksize, col:col + ksize, :]
                patch_tuple["opt_t0"] = opt_t0[row:row + ksize, col:col + ksize, :]
                patch_tuple["sar_t1"] = sar_t1[row:row + ksize, col:col + ksize, :]
                patch_tuple["opt_t1"] = opt_t1[row:row + ksize, col:col + ksize, :]
                patch_tuple["labels"] = labels[row:row + ksize, col:col + ksize]
                np.save(output_folder + str(cont), patch_tuple)
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
                cont += 1
    print (m2.shape)
    plt.figure()
    plt.imshow(m2)
    plt.show(block=True)
    return 0



def extract_patches_4_classifierQuemadas(
        ksize,
        mask,
        mask_sar,
        labels,
        output_folder,
        sar,
        opt,
        stride=128,
        show=False):

    print (output_folder)
    rows, cols = mask.shape
    mask_val = load_tiff_image('quemadas_samples4val_maks.tif')
    mask_val = mask_val[0, :rows, :cols]
    mask_val[mask_val != 0] = 1
    print(mask_val.shape)
    plt.figure()
    plt.imshow(mask_val)
    plt.show(block=False)
    cont = 0
    m2 = mask.copy()
    tst_index = []
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) > (ksize**2) / 1.1) and (np.sum(auxmask==2) == 0) and (np.sum(auxmask==1) > 50):
                print ("Patch valido ...", cont)
                patch_tuple = {}
                patch_sar = sar[row:row + ksize, col:col + ksize, :]
                patch_tuple["sar"] = patch_sar
                patch_opt = opt[row:row + ksize, col:col + ksize, :]
                patch_tuple["opt"] = patch_opt
                patch_labels = labels[row:row + ksize, col:col + ksize]
                patch_tuple["labels"] = patch_labels
                # patch_tuple = np.concatenate((patch_sar, patch_opt), axis=2)
                # np.save(output_folder + str(cont), patch_tuple)
                # Load read_dictionary = np.load('my_file.npy').item()
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
                if (np.sum(mask_val[row:row + ksize, col:col + ksize]) >= 256**2/1.5):
                    print ("Validation sample -->", cont)
                    if not os.path.exists(output_folder + 'Validation/'):
                        os.makedirs(output_folder + 'Validation/')
                    np.save(output_folder + 'Validation/' + str(cont), patch_tuple)
                else:
                    print ("trainig sample -->", cont)
                    if not os.path.exists(output_folder + 'Training/'):
                        os.makedirs(output_folder + 'Training/')
                    np.save(output_folder + 'Training/' + str(cont), patch_tuple)
                cont += 1
                
                
    # m2[mask==2] = 0
    # figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi
    # my_dpi = 300
    print (m2.shape)
    plt.figure()
    # plt.figure(figsize=(m2.shape[1]/my_dpi, m2.shape[0]/my_dpi), dpi=my_dpi)
    plt.imshow(m2)
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig('quemadas_samples.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show(block=True)
    # train_area = m2[m2!=0]
    # print(train_area.shape)
    # print ()
    return 0

def extract_patches_stride_case_A(
        ksize,
        mask,
        mask_sar,
        output_folder,
        sar,
        opt,
        stride=35,
        show=False):

    rows, cols = mask.shape
    cont = 0
    m2 = mask.copy()
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) > (ksize**2) / 1.1) and (np.sum(auxmask==0) < (ksize**2) / 1.2) and (np.sum(auxmask==2) == 0):
                print ("Patch valido ...", cont)
                patch_sar = sar[row:row + ksize, col:col + ksize, :]
                patch_opt = opt[row:row + ksize, col:col + ksize, :]
                patch_par = np.concatenate((patch_sar, patch_opt), axis=2)
                # np.save(output_folder + str(cont), patch_par)
                cont += 1
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
    # m2[mask==2] = 0
    plt.imshow(m2)
    plt.show()
    # train_area = m2[m2!=0]
    # print(train_area.shape)
    # print ()
    return 0


def extract_patches_stride3(ksize,
                           mask,
                           mask_sar,
                           output_folder,
                           sar,
                           opt,
                           show=False):
    rows, cols = mask.shape
    cont = 0
    m2 = mask.copy()
    for row in range(0, rows-ksize-1, 23):
        for col in range(0, cols-ksize-1, 23):
#            print ("Patch  ...", np.sum(mask[row:row+ksize, col:col+ksize]))
            auxmask = mask[row:row+ksize, col:col+ksize]
            if (np.sum(mask_sar[row:row+ksize, col:col+ksize]) > 10) and (np.sum(auxmask==0) < (ksize**2)/1.1) and (np.sum(auxmask==2) == 0):
                print ("Patch valido ...", cont)
                patch_sar = sar[row:row+ksize, col:col+ksize, :]
                patch_opt = opt[row:row+ksize, col:col+ksize, :]
                patch_par = np.concatenate((patch_sar, patch_opt), axis=2)
                np.save(output_folder + str(cont), patch_par)
                cont += 1
                m2[row:row+ksize, col:col+ksize] = np.random.randint(255)
    plt.imshow(m2)
    plt.show()
    return 0


def extract_patches_stride_multiresolution(ksize,
                                           mask,
                                           mask_sar,
                                           output_folder,
                                           sar,
                                           opt,
                                           show=False):
    rows, cols = mask.shape
    cont = 0
    m2 = mask.copy()
    for row in range(0, rows-ksize-1, 45):
        for col in range(0, cols-ksize-1, 45):
#            print ("Patch  ...", np.sum(mask[row:row+ksize, col:col+ksize]))
            auxmask = mask[row:row+ksize, col:col+ksize]
            if (np.sum(mask_sar[row:row+ksize, col:col+ksize]) < 10) and (np.sum(auxmask==0) < (ksize**2)/1.1) and (np.sum(auxmask==2) == 0):
                print ("Patch valido ...", cont)
                print ("mask ...", np.sum(auxmask==0))

                patch_sar = sar[row:row+ksize, col:col+ksize, :]
                patch_opt = opt[row//3:row//3+ksize//3, col//3:col//3+ksize//3, :]
                patch_par = [patch_sar, patch_opt]
#                patch_par = np.concatenate((patch_sar, patch_opt), axis=2)
                np.save(output_folder + str(cont), patch_par)
                cont += 1
                m2[row:row+ksize, col:col+ksize] = np.random.randint(255)
    plt.figure()
    plt.imshow(m2)
    plt.show()
    return 0


def extract_patches_stride_multiresolutionx3(ksize,
                                             mask,
                                             mask_sar,
                                             cloud_mask,
                                             output_folder,
                                             sar,
                                             img_landsat,
                                             img_sent2,
                                             show=True):
    rows, cols = mask.shape
    cont = 0
    m2 = mask.copy()
    for row in range(0, rows-ksize-1, 55):
        for col in range(0, cols-ksize-1, 55):
#            print ("Patch  ...", np.sum(mask[row:row+ksize, col:col+ksize]))
            auxmask = mask[row:row+ksize, col:col+ksize]
            auxcloud = cloud_mask[row:row+ksize, col:col+ksize]
            if (np.sum(mask_sar[row:row+ksize, col:col+ksize]) > (ksize**2)/1.1) and (np.sum(auxmask==0) < (ksize**2)/1.1) and (np.sum(auxmask==2) == 0) and (np.sum(auxcloud == 1) < 5):
                print ("Patch valido ...", cont)
                print ("mask ...", np.sum(auxmask==0))

                patch_sar = sar[row:row+ksize, col:col+ksize, :]
                patch_sent2 = img_sent2[row:row+ksize, col:col+ksize, :]
                patch_landsat = img_landsat[row//3:row//3+ksize//3, col//3:col//3+ksize//3, :]
                patches = [patch_sar, patch_sent2, patch_landsat]
#                patch_par = np.concatenate((patch_sar, patch_opt), axis=2)
                np.save(output_folder + str(cont), patches)
                cont += 1
                m2[row:row+ksize, col:col+ksize] = np.random.randint(255)
#                plt.figure(3)
#                plt.imshow(patch_sar[:, :, 1], cmap='gray')
#                plt.show(block=False)
#                plt.pause(0.5)
#                plt.figure(4)
#                plt.imshow(patch_sent2[:, :, 0], cmap='gray')
#                plt.show(block=False)
#                plt.pause(0.5)
#                plt.figure(5)
#                plt.imshow(patch_landsat[:, :, 0], cmap='gray')
#                plt.show(block=False)
#                plt.pause(0.5)
    plt.figure()
    plt.imshow(m2)
    plt.show()
    return 0


def save_imgs_pars_stride(output_folder, sar, opt, ksize, row, col, cont=0):

    patch_sar = get_patch(sar, ksize, row, col)
    patch_opt = get_patch(opt, ksize, row, col)
    patch_par = np.concatenate((patch_sar, patch_opt), axis=2)
    np.save(output_folder + str(cont), patch_par)
    return patch_par


def mask_4_trn_gans(num_rows, num_cols, region='all'):

    if region is 'top':
        mask_gans_trn = np.ones((num_rows, num_cols))
        mask_gans_trn[:int(num_rows/2), :num_cols] = 0

    elif region is 'bottom':
        mask_gans_trn = np.zeros((num_rows, num_cols))
        mask_gans_trn[:int(num_rows/2), :num_cols] = 1

    elif region is 'all':
        mask_gans_trn = np.ones((num_rows, num_cols))
    else:
        print "kabooom"
    return mask_gans_trn


def minmaxnormalization(img, mask, scaler_name="sample.pkl"):

    num_rows, num_cols, bands = img.shape
    img = img.reshape(num_rows * num_cols, bands)
    # print(img.max(), img.min())
    # print img.shape
    scaler = pre.MinMaxScaler((-1, 1)).fit(img[mask.ravel() == 1])
    # save model
    joblib.dump(scaler, scaler_name)
    # del scaler
    # scaler = joblib.load("opt_05may2016_scaler.pkl")
    img = np.float32(scaler.transform(img))
    img = img.reshape(num_rows, num_cols, bands)

    return img


def create_dataset_case_A(ksize=256,
                          dataset=None,
                          mask_path=None,
                          sar_path=None,
                          opt_path=None
                          ):

    # patch_trn = '/mnt/Data/Pix2Pix_datasets/Semi_Exp/' + dataset + '/Classifier/train/'
    patch_trn = '/mnt/Data/Pix2Pix_datasets/Semi_Exp/' + dataset + '/train/'
    root_path = '/mnt/Data/DataBases/RS/'
    labels_path = root_path + 'Quemadas/AP2_Acre/clip_reference_raster_new.tif'
    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)

    labels = load_tiff_image(labels_path)
    labels[np.isnan(labels)] = 0
    labels[labels != 0] = 1

    opt = load_sentinel2(opt_path)
    opt[np.isnan(opt)] = 0
    print opt.shape
    sar = np.load(sar_path)
    mask_gans_trn = load_tiff_image('new_train_test_mask.tif')
    # mask_gans_trn = 'ap2_train_test_mask.npy'
    # mask_gans_trn = np.load(mask_gans_trn)
    mask_gans_trn = np.float32(mask_gans_trn)
    mask_gans_trn[mask_gans_trn == 0] = 1.
    mask_gans_trn[mask_gans_trn == 255] = 2.
    print mask_gans_trn.shape

    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 1].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0

    mask_gans_trn = mask_gans_trn * mask_sar
    mask_opt = mask_gans_trn.copy()
    mask_gans_trn[mask_sar == 0] = 2
    mask_gans_trn[mask_gans_trn == 1] = 0.
    # mask_gans_trn = mask_gans_trn + labels
    mask_gans_trn = mask_gans_trn
    ########MODIFICAR ESTO ##############

    plt.figure()
    plt.imshow(mask_opt)
    plt.show(block=False)
    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    plt.figure()
    plt.imshow(mask_sar)
    plt.show(block=False)
    # plt.close('all')

    sar = minmaxnormalization(sar, mask_sar)
    opt = minmaxnormalization(opt, mask_opt)

    extract_patches_4_testing(
        ksize=ksize,
        mask=mask_gans_trn,
        mask_sar=mask_sar,
        labels=labels,
        output_folder=patch_trn,
        sar=sar,
        opt=opt,
        stride=256,
        show=True)

    # extract_patches_4_classifierQuemadas(
    #     ksize=ksize,
    #     mask=mask_gans_trn,
    #     mask_sar=mask_sar,
    #     labels=labels,
    #     output_folder=patch_trn,
    #     sar=sar,
    #     opt=opt,
    #     stride=226,
    #     show=True)

    # extract_patches_stride_case_A(
    #     ksize,
    #     mask=mask_gans_trn,
    #     mask_sar=mask_sar,
    #     output_folder=patch_trn,
    #     sar=sar,
    #     opt=opt,
    #     stride=226,
    #     show=True)


def create_dataset_Quemandas_Multitemporal(ksize=128,
                                           dataset=None,
                                           mask_path=None,
                                           sar_path_t0=None,
                                           opt_path_t0=None,
                                           sar_path_t1=None,
                                           opt_path_t1=None,
                                           show=False):
    
    output_folder = '/mnt/Data/Pix2Pix_datasets/Quemadas/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    root_path = '/mnt/Data/DataBases/RS/'
    labels_path = root_path + 'Quemadas/AP2_Acre/clip_reference_raster_new.tif'

    labels = load_tiff_image(labels_path)
    labels[np.isnan(labels)] = 0
    labels[labels != 0] = 1

    opt_t0 = load_sentinel2(opt_path_t0)
    opt_t1 = load_sentinel2(opt_path_t1)
    opt_t0[np.isnan(opt_t0)] = 0
    opt_t1[np.isnan(opt_t1)] = 0
    print opt_t0.shape
    print opt_t1.shape
    sar_t0 = np.load(sar_path_t0)
    sar_t1 = np.load(sar_path_t1)
    mask_gans_trn = load_tiff_image('new_train_test_mask.tif')
    # mask_gans_trn = 'ap2_train_test_mask.npy'
    # mask_gans_trn = np.load(mask_gans_trn)
    mask_gans_trn = np.float32(mask_gans_trn)
    mask_gans_trn[mask_gans_trn == 0] = 1.
    mask_gans_trn[mask_gans_trn == 255] = 2.
    print mask_gans_trn.shape

    sar_t0[sar_t0 > 1.0] = 1.0
    sar_t1[sar_t1 > 1.0] = 1.0
    mask_sar_t0 = sar_t0[:, :, 1].copy()
    mask_sar_t0[sar_t0[:, :, 0] < 1] = 1
    mask_sar_t0[sar_t0[:, :, 0] == 1] = 0
    mask_sar_t1 = sar_t1[:, :, 1].copy()
    mask_sar_t1[sar_t1[:, :, 0] < 1] = 1
    mask_sar_t1[sar_t1[:, :, 0] == 1] = 0
    mask_sar = mask_sar_t0 * mask_sar_t1

    mask_gans_trn = mask_gans_trn * mask_sar
    mask_opt = mask_gans_trn.copy()
    mask_gans_trn[mask_sar == 0] = 2
    mask_gans_trn[mask_gans_trn == 1] = 0.
    mask_gans_trn = mask_gans_trn + labels

    plt.figure()
    plt.imshow(mask_opt)
    plt.show(block=False)
    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    plt.figure()
    plt.imshow(mask_sar)
    plt.show(block=False)

    sar_t0 = minmaxnormalization(sar_t0, mask_sar, output_folder + 'sar_t0_Scaler')
    sar_t1 = minmaxnormalization(sar_t1, mask_sar, output_folder + 'sar_t1_Scaler')
    opt_t0 = minmaxnormalization(opt_t0, mask_opt, output_folder + 'opt_t0_Scaler')
    opt_t1 = minmaxnormalization(opt_t1, mask_sar, output_folder + 'opt_t1_Scaler')

    extract_patches_multitemporal_Quemandas(
        ksize=128, # cambiar despues
        mask=mask_gans_trn,
        mask_sar=mask_sar,
        output_folder=output_folder,
        sar_t0=sar_t0,
        sar_t1=sar_t1,
        opt_t0=opt_t0,
        opt_t1=opt_t1,
        labels=labels,
        stride=51,
        show=True)


def create_dataset_case3(ksize=256,
                         dataset=None,
                         mask_path=None,
                         sar_path=None,
                         opt_path=None,
                         region=None,
                         num_patches=400,
                         show=False):
    patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/'
    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)
    mask_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/New_Masks/TrainTestMasks/TrainTestMask_50_50_Dec.tif'
    opt = load_sentinel2(opt_path)
    opt[np.isnan(opt)] = 0
    sar = np.load(sar_path)
#    sar = np.rollaxis(sar, 0, 3)
    mask = load_tiff_image(mask_path)
    mask_gans_trn = mask.copy()
    mask_gans_trn[mask_gans_trn!=0] = 1

    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 1].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0


    plt.figure()
    plt.imshow(mask_sar)
    plt.show(block=True)

    sar = minmaxnormalization(sar, mask_sar)
    opt = minmaxnormalization(opt, mask_sar)

    extract_patches_stride3(ksize,
                           mask=mask_gans_trn,
                           mask_sar=mask_sar,
                           output_folder=patch_trn,
                           sar=sar,
                           opt=opt,
                           show=True)


def create_dataset_case4(ksize=256,
                         dataset=None,
                         mask_path=None,
                         sar_path=None,
                         opt_path=None,
                         region=None,
                         num_patches=400,
                         show=False):
    patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/'
    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)
    mask_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/New_Masks/TrainTestMasks/TrainTestMask_50_50_Dec.tif'
    opt, _ = load_landsat(opt_path)
    opt[np.isnan(opt)] = 0.0
    opt = np.float32(opt)
    sar = np.load(sar_path)
    sar = np.rollaxis(sar, 0, 3)
    mask = load_tiff_image(mask_path)

    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 0].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0

    mask_gan = np.load('mask_gan_original.npy')
    mask_gan[mask == 0] = 0
    mask_gan[mask_gan != 0] = 2
    mask_gan[(mask != 0) * (mask_gan == 0)] = 1

    mask_gans_trn = mask_gan

    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    plt.figure()
    plt.imshow(mask_sar)
    plt.show(block=False)

    sar = minmaxnormalization(sar, mask_sar)
    mask_opt = resampler(mask_sar)
    print opt.shape
    print mask_opt.shape
    plt.figure()
    plt.imshow(mask_opt)
    plt.show(block=False)
    opt = minmaxnormalization(opt, mask_opt)
#    sar2 = resampler(sar)

    extract_patches_stride_multitemporal(ksize=3*ksize,
                                         mask=mask_gans_trn,
                                         mask_sar=mask_sar,
                                         output_folder=patch_trn,
                                         sar=sar,
                                         opt=opt,
                                         show=True)


def create_dataset_case5(ksize=256,
                         dataset=None,
                         mask_path=None,
                         sar_path=None,
                         opt_path=None,
                         region=None,
                         num_patches=400,
                         show=False):
    patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/'
    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)
    mask_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/New_Masks/TrainTestMasks/TrainTestMask_50_50_Dec.tif'
    opt, _ = load_landsat(opt_path)
    opt[np.isnan(opt)] = 0.0
    opt = np.float32(opt)
    sar = np.load(sar_path)
#    sar = np.rollaxis(sar, 0, 3)
    mask = load_tiff_image(mask_path)
    mask = resampler(mask)

    sar = resampler(sar)
    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 0].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0

    mask_gan = np.load('mask_gan_original.npy')
    mask_gan = resampler(mask_gan)
    mask_gan[mask == 0] = 0
    mask_gan[mask_gan != 0] = 2
    mask_gan[(mask != 0) * (mask_gan == 0) ] = 1

    mask_gans_trn = mask_gan

    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    plt.figure()
    plt.imshow(mask_sar)
    plt.show(block=False)

    sar = minmaxnormalization(sar, mask_gans_trn)
    print opt.shape
    plt.figure()
    plt.imshow(mask_sar)
    plt.show(block=False)
    opt = minmaxnormalization(opt, mask_gans_trn)

    extract_patches_stride3(ksize=ksize,
                            mask=mask_gans_trn,
                            mask_sar=mask_sar,
                            output_folder=patch_trn,
                            sar=sar,
                            opt=opt,
                            show=True)



def create_dataset_multitemporal_multiresolution_CV(ksize=256,
                                                    dataset=None,
                                                    mask_path=None,
                                                    sar_path_t0=None,
                                                    opt_path_t0=None,
                                                    sar_path_t1=None,
                                                    opt_path_t1=None):

    # patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/classifier/'
    output_folder = '/mnt/Data/Pix2Pix_datasets/Campo_Verde_50_50/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mask, sar_t0, opt_t0, _ = load_images(mask_path=None,
                                           sar_path=sar_path_t0,
                                           opt_path=opt_path_t0)
    _, sar_t1, opt_t1, _ = load_images(mask_path=None,
                                       sar_path=sar_path_t1,
                                       opt_path=opt_path_t1)
    opt_t1[np.isnan(opt_t1)] = 0.0
    opt_t0[np.isnan(opt_t0)] = 0.0
    labels_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/'
    labels_name = '10_May_2016.tif'
    labels = load_tiff_image(labels_root_path + labels_name)
    labels = resampler(labels, 'uint8')

    sar_t0[sar_t0 > 1.0] = 1.0
    sar_t1[sar_t1 > 1.0] = 1.0
    mask_sar0 = sar_t0[:, :, 0].copy()
    mask_sar0[sar_t0[:, :, 0] < 1] = 1
    mask_sar0[sar_t0[:, :, 0] == 1] = 0
    mask_sar1 = sar_t1[:, :, 0].copy()
    mask_sar1[sar_t1[:, :, 0] < 1] = 1
    mask_sar1[sar_t1[:, :, 0] == 1] = 0
    mask_sar = mask_sar1 * mask_sar0 # Define mascara SAR para las dos imagenes
    mask_sar4opt = resampler(mask_sar, 'uint8') # Crea una mascara SAR en la resolucion de 30m.
    plt.figure('mask sar')
    plt.imshow(mask_sar)
    plt.show(block=False)
    mask_opt0 = opt_t0[:, :, 3].copy()
    mask_opt0[mask_opt0 != 0] = 1
    mask_opt1 = opt_t1[:, :, 3].copy()
    mask_opt1[mask_opt1 != 0] = 1
    mask_opt = mask_opt1 * mask_opt0 # Define la mascara general para las imagenes opticas.
    plt.figure('mask opt')
    plt.imshow(mask_opt)
    plt.show(block=False)

    mask4opt = resampler(mask, 'uint8')
    plt.figure('mask4opt')
    plt.imshow(mask4opt)
    plt.show(block=False)
#    mask_gan = resampler(mask_gan)
    mask_gan = np.load('mask_gan.npy')
    # mask_gan[mask == 0] = 0
    plt.figure('labels')
    plt.imshow(labels)
    plt.show(block=False)
    labels_trn = labels.copy()
    # print ("max-min optical image ----> ",opt.max(), opt.min())
    labels_trn[(mask4opt != 0) * (mask_gan != 1)] = 0
    plt.figure('labels_trn')
    plt.imshow(labels_trn)
    plt.show(block=False)
    mask_gan[(mask4opt != 0) * (mask_gan != 1)] = 2
    # mask_opt = opt[:, :, 3].copy()
    # mask_opt[mask_opt != 0] = 1

    # mask_gans_trn = mask_gan * mask_sar * mask_opt
    mask_gans_trn_opt = mask_sar4opt * mask_opt
    plt.figure('mask_gans_trn_opt')
    plt.imshow(mask_gans_trn_opt)
    plt.show(block=False)
    mask_gans_trn_sar = up_sampler(mask_gans_trn_opt, 'uint8')
    plt.figure('mask_gans_trn_sar')
    plt.imshow(mask_gans_trn_sar)
    plt.show(block=False)


    mask_gans2 = np.load('mask_gans_trn.npy')
    mask_gans2[mask4opt == 0] = 0
    plt.figure('mask_gans')
    plt.imshow(mask_gans2)
    plt.show(block=False)

    mask_gans_opt_t0 = mask_gans_trn_opt.copy()
    mask_gans_opt_t0[mask_gan == 2] = 0
    plt.figure('mask_gans_opt_t0')
    plt.imshow(mask_gans_opt_t0)
    plt.show(block=False)

    sar_t0 = minmaxnormalization(sar_t0, mask_gans_trn_sar, scaler_name='sar_may2016_10m_scaler.pkl')
    opt_t0 = minmaxnormalization(opt_t0, mask_gans_opt_t0, scaler_name='opt_may2016_scaler.pkl')
    sar_t1 = minmaxnormalization(sar_t1, mask_gans_trn_sar, scaler_name='sar_may2017_10m_scaler.pkl')
    opt_t1 = minmaxnormalization(opt_t1, mask_gans_trn_opt, scaler_name='opt_may2017_scaler.pkl')
    # plt.show(block=True)
    # img_A = np.concatenate((sar_t0, sar_t1, opt_t1), axis=2)
    # img_B = opt_t0
    # np.save('mask_gans_trn.npy', mask_gans_trn)
    # mask_gans_trn = np.load('mask_gans_trn.npy') descomentar para la version original
    # mask_gans_trn[mask == 0] = 0
    # plt.figure('Mask 2')
    # plt.imshow(mask_gans_trn)
    # plt.show(block=False)
    # print ("max-min normalized optical image ----> ", opt[mask_gans_trn==1].max(), opt[mask_gans_trn==1].min())
    # print ("max-min normalized sar image ----> ", sar[mask_gans_trn==1].max(), sar[mask_gans_trn==1].min())

    # mask_gan = np.load('mask_gans_trn.npy')
    # mask_gan[mask == 0] = 0
    # plt.figure('Mask Gan')
    # plt.imshow(mask_gan)
    # plt.show(block=False)

    extract_patches_multitemporal_multiresolution(
        ksize=ksize//4,
        mask=mask4opt,
        mask_sar=mask_gans_trn_opt,
        labels=labels_trn,
        output_folder=output_folder,
        sar_t0=sar_t0,
        opt_t0=opt_t0,
        sar_t1=sar_t1,
        opt_t1=opt_t1,
        # stride=ksize // 17,
        stride=ksize // 4,
        block=True)

    # mask_gans_trn = np.load('mask_gans_trn.npy')
    # mask_gans_trn[mask == 0] = 0
    # extract_patches_4_testing_multitemporal(
    #     ksize=ksize,
    #     mask=mask_gans2,
    #     mask_sar=mask_gans_trn_opt,
    #     labels=labels_trn,
    #     output_folder=output_folder,
    #     sar_t0=sar_t0,
    #     opt_t0=opt_t0,
    #     sar_t1=sar_t1,
    #     opt_t1=opt_t1,
    #     stride=ksize // 4,
    #     block=True)
    # return 0


def create_dataset_4_classifier_multitemporal(ksize=256,
                                              dataset=None,
                                              mask_path=None,
                                              sar_path_t0=None,
                                              opt_path_t0=None,
                                              sar_path_t1=None,
                                              opt_path_t1=None):

    # patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/classifier/'
    output_folder = '/mnt/Data/Pix2Pix_datasets/Semi_Exp/' + dataset + '/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mask, sar_t0, opt_t0, _ = load_images(mask_path=None,
                                                   sar_path=sar_path_t0,
                                                   opt_path=opt_path_t0)
    _, sar_t1, opt_t1, _ = load_images(mask_path=None,
                                       sar_path=sar_path_t1,
                                       opt_path=opt_path_t1)
    opt_t1[np.isnan(opt_t1)] = 0.0
    opt_t0[np.isnan(opt_t0)] = 0.0
    labels_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/'
    labels_name = '10_May_2016.tif'
    labels = load_tiff_image(labels_root_path + labels_name)
    labels = resampler(labels, 'uint8')

    sar_t0 = resampler(sar_t0, 'float32')
    sar_t1 = resampler(sar_t1, 'float32')
    sar_t0[sar_t0 > 1.0] = 1.0
    sar_t1[sar_t1 > 1.0] = 1.0
    mask_sar0 = sar_t0[:, :, 0].copy()
    mask_sar0[sar_t0[:, :, 0] < 1] = 1
    mask_sar0[sar_t0[:, :, 0] == 1] = 0
    mask_sar1 = sar_t1[:, :, 0].copy()
    mask_sar1[sar_t1[:, :, 0] < 1] = 1
    mask_sar1[sar_t1[:, :, 0] == 1] = 0
    mask_sar = mask_sar1 * mask_sar0
    plt.figure('mask sar')
    plt.imshow(mask_sar)
    plt.show(block=False)
    mask_opt0 = opt_t0[:, :, 3].copy()
    mask_opt0[mask_opt0 != 0] = 1
    mask_opt1 = opt_t1[:, :, 3].copy()
    mask_opt1[mask_opt1 != 0] = 1
    mask_opt = mask_opt1 * mask_opt0
    plt.figure('mask opt')
    plt.imshow(mask_opt)
    plt.show(block=False)

    mask = resampler(mask, 'uint8')
#    mask_gan = resampler(mask_gan)
    mask_gan = np.load('mask_gan.npy')
    # mask_gan[mask == 0] = 0
    plt.figure('labels')
    plt.imshow(labels)
    plt.show(block=False)
    labels_trn = labels.copy()
    # print ("max-min optical image ----> ",opt.max(), opt.min())
    labels_trn[(mask != 0) * (mask_gan != 1)] = 0
    plt.figure('labels_trn')
    plt.imshow(labels_trn)
    plt.show(block=False)
    mask_gan[(mask != 0) * (mask_gan != 1)] = 2
    # mask_opt = opt[:, :, 3].copy()
    # mask_opt[mask_opt != 0] = 1

    # mask_gans_trn = mask_gan * mask_sar * mask_opt
    mask_gans_trn = mask_sar * mask_opt
    plt.figure('mask_gans_trn')
    plt.imshow(mask_gans_trn)
    plt.show(block=False)

    mask_gans2 = np.load('mask_gans_trn.npy')
    mask_gans2[mask == 0] = 0
    plt.figure('mask_gans')
    plt.imshow(mask_gans2)
    plt.show(block=False)

    mask_gans_opt_t0 = mask_gans_trn.copy()
    mask_gans_opt_t0[mask_gan == 2] = 0
    plt.figure('mask_gans_opt_t0')
    plt.imshow(mask_gans_opt_t0)
    plt.show(block=False)

    sar_t0 = minmaxnormalization(sar_t0, mask_gans_trn, scaler_name='sar_may2016_scaler.pkl')
    opt_t0 = minmaxnormalization(opt_t0, mask_gans_opt_t0, scaler_name='opt_may2016_scaler.pkl')
    sar_t1 = minmaxnormalization(sar_t1, mask_gans_trn, scaler_name='sar_may2017_scaler.pkl')
    opt_t1 = minmaxnormalization(opt_t1, mask_gans_trn, scaler_name='opt_may2017_scaler.pkl')
    img_A = np.concatenate((sar_t0, sar_t1, opt_t1), axis=2)
    img_B = opt_t0
    # np.save('mask_gans_trn.npy', mask_gans_trn)
    # mask_gans_trn = np.load('mask_gans_trn.npy') descomentar para la version original
    # mask_gans_trn[mask == 0] = 0
    # plt.figure('Mask 2')
    # plt.imshow(mask_gans_trn)
    # plt.show(block=False)
    # print ("max-min normalized optical image ----> ", opt[mask_gans_trn==1].max(), opt[mask_gans_trn==1].min())
    # print ("max-min normalized sar image ----> ", sar[mask_gans_trn==1].max(), sar[mask_gans_trn==1].min())

    # mask_gan = np.load('mask_gans_trn.npy')
    # mask_gan[mask == 0] = 0
    # plt.figure('Mask Gan')
    # plt.imshow(mask_gan)
    # plt.show(block=False)

    # extract_patches_4_classifier_multitemporal(
    #     ksize=ksize,
    #     mask=mask_gans2,
    #     mask_sar=mask_gans_trn,
    #     labels=labels_trn,
    #     output_folder=output_folder,
    #     img_A=img_A,
    #     img_B=img_B,
    #     stride=ksize // 4,
    #     block=False)

    # # mask_gans_trn = np.load('mask_gans_trn.npy')
    # # mask_gans_trn[mask == 0] = 0
    # extract_patches_4_testing_multitemporal(
    #     ksize=ksize,
    #     mask=mask_gan,
    #     mask_sar=mask_gans_trn,
    #     labels=labels,
    #     output_folder=output_folder,
    #     img_A=img_A,
    #     img_B=img_B,
    #     stride=ksize // 2,
    #     block=True)
    return 0


def create_dataset_4_classifier(ksize=256,
                                dataset=None,
                                mask_path=None,
                                sar_path=None,
                                opt_path=None,
                                region=None,
                                num_patches=400,
                                show=False):

    # patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/classifier/'
    output_folder = '/mnt/Data/Pix2Pix_datasets/Semi_Exp/' + dataset + '/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mask, sar, opt, cloud_mask = load_images(mask_path=None,
                                             sar_path=sar_path,
                                             opt_path=opt_path)
    labels_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/'
    labels_name = '10_May_2016.tif'
    labels = load_tiff_image(labels_root_path + labels_name)
    labels = resampler(labels, 'uint8')

    cloud_mask[cloud_mask != 0] = 1

    sar = resampler(sar, 'float32')
    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 0].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0
    mask = resampler(mask, 'uint8')
#    mask_gan = resampler(mask_gan)
    mask_gan = np.load('mask_gan.npy')
    # mask_gan[mask == 0] = 0
    plt.figure('labels')
    plt.imshow(labels)
    plt.show(block=False)
    labels_trn = labels.copy()
    print ("max-min optical image ----> ",opt.max(), opt.min())
    labels_trn[(mask != 0) * (mask_gan != 1)] = 0
    plt.figure('labels_trn')
    plt.imshow(labels_trn)
    plt.show(block=False)
    mask_gan[(mask != 0) * (mask_gan != 1)] = 2
    mask_opt = opt[:, :, 3].copy()
    mask_opt[mask_opt != 0] = 1

    # mask_gans_trn = mask_gan * mask_sar * mask_opt
    mask_gans_trn = mask_sar * mask_opt
    plt.figure('mask_gans_trn')
    plt.imshow(mask_gans_trn)
    plt.show(block=False)

    sar = minmaxnormalization(sar, mask_gans_trn, scaler_name='sar_05may2016_scaler.pkl')
    opt = minmaxnormalization(opt, mask_gans_trn, scaler_name='opt_05may2016_scaler.pkl')
    # np.save('mask_gans_trn.npy', mask_gans_trn)
    # mask_gans_trn = np.load('mask_gans_trn.npy') descomentar para la version original
    # mask_gans_trn[mask == 0] = 0
    # plt.figure('Mask 2')
    # plt.imshow(mask_gans_trn)
    # plt.show(block=False)
    print ("max-min normalized optical image ----> ", opt[mask_gans_trn==1].max(), opt[mask_gans_trn==1].min())
    print ("max-min normalized sar image ----> ", sar[mask_gans_trn==1].max(), sar[mask_gans_trn==1].min())

    extract_patches_4_classifier(
        ksize=ksize,
        mask=mask_gans_trn,
        mask_sar=mask_sar,
        labels=labels_trn,
        output_folder=output_folder,
        sar=sar,
        opt=opt,
        stride=256 // 3)

    mask_gans_trn = np.load('mask_gans_trn.npy')
    mask_gans_trn[mask == 0] = 0
    extract_patches_4_testing(
        ksize=ksize,
        mask=mask_gans_trn,
        mask_sar=mask_sar,
        labels=labels,
        output_folder=output_folder,
        sar=sar,
        opt=opt,
        stride=200,
        show=True)
    return 0


def create_dataset_case1(ksize=256,
                         dataset=None,
                         mask_path=None,
                         sar_path=None,
                         opt_path=None,
                         region=None,
                         num_patches=400,
                         show=False):
    patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/'

    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)

    mask, sar, opt, cloud_mask = load_images(mask_path=None,
                                             sar_path=sar_path,
                                             opt_path=opt_path)
    labels_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/'
    labels_name = '10_May_2016.tif'
    labels = load_tiff_image(labels_root_path + labels_name)
    labels = resampler(labels)
#    print cloud_mask.max(), cloud_mask.min()
    cloud_mask[cloud_mask != 0] = 1
#    mask_gan_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/New_Masks/TrainTestMasks/TrainTestMask_GAN.tif'
#    mask_gan = load_tiff_image(mask_gan_path)
#    mask_gan[mask_gan == 0] = 1
#    mask_gan[mask_gan != 1] = 0
#    print mask_gan.max(), mask_gan.min()

#    plt.figure(1)
#    plt.imshow(cloud_mask)
#    plt.show()
    sar = resampler(sar)
    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 0].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0
    mask = resampler(mask)
    mask_gan = np.load('mask_gan.npy')
    mask_gan[mask == 0] = 0
    mask_gan[(mask != 0) * (mask_gan != 1)] = 2
    mask_gans_trn = mask_gan * mask_sar
    # mask_gans_trn[mask_gans_trn == 3] = 0
    # mask_gans_trn[mask_gans_trn == 2] = 1
    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    # plt.figure()
    # plt.imshow(cloud_mask)
    # plt.show(block=True)

    sar = minmaxnormalization(sar, mask_sar)
    opt = minmaxnormalization(opt, mask_gans_trn)

    # extract_patches_4_classifier(
    #     ksize=ksize,
    #     mask=mask_gans_trn,
    #     mask_sar=mask_sar,
    #     labels=labels,
    #     output_folder=patch_trn,
    #     sar=sar,
    #     opt=opt)

    extract_patches_stride_case_A(
        ksize=ksize,
        mask=mask_gans_trn,
        mask_sar=mask_sar,
        output_folder=patch_trn,
        sar=sar,
        opt=opt)

#    extract_patches_stride(ksize,
#                           mask=mask_gans_trn,
#                           cloud_mask=cloud_mask,
#                           output_folder=patch_trn,
#                           sar=sar,
#                           opt=opt,
#                           show=True)
#    extract_patches(ksize,
#                    mask=mask_gans_trn,
#                    cloud_mask=cloud_mask,
#                    output_folder=patch_trn,
#                    sar=sar,
#                    opt=opt,
#                    num_patches=num_patches,
#                    show=True)
#    row_trn, col_trn, row_tst, col_tst = get_pixel_index(ksize,
#                                                         mask_gans_trn,
#                                                         num_patches,
#                                                         show=True)

#    save_imgs_pars(patch_trn, sar, opt, ksize, row_trn, col_trn)
#    save_imgs_pars(patch_tst, sar, opt, ksize, row_tst, col_tst)
#    print np.concatenate((row_trn.reshape(len(row_trn), 1),
#                          col_trn.reshape(len(row_trn), 1)), axis=1)

#    return patch_par_trn

def create_dataset_case2(ksize=256,
                         dataset=None,
                         mask_path=None,
                         sar_path1=None,
                         sar_path2=None,
                         opt_path1=None,
                         opt_path2=None,
                         region=None):
    patch_trn = '/home/lvc/Experiments/Pix2pix/datasets/'+dataset+'/train/'
    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)
    mask, sar1, opt_target, cloud_mask1 = load_images(mask_path=None,
                                                      sar_path=sar_path1,
                                                      opt_path=opt_path1)
    _, sar2, opt_source, cloud_mask2 = load_images(mask_path=None,
                                                   sar_path=sar_path2,
                                                   opt_path=opt_path2)
    cloud_mask = cloud_mask1+cloud_mask2
    sar1 = resampler(sar1)
    sar2 = resampler(sar2)
    mask = resampler(mask)
    sar1[sar1 > 1.0] = 1.0
    sar2[sar2 > 1.0] = 1.0
    mask_sar = sar1[:, :, 0].copy()
    mask_sar[sar1[:, :, 0] < 1] = 1
    mask_sar[sar1[:, :, 0] == 1] = 0
    mask_gan = np.load('mask_gan_original.npy')
    mask_gan = resampler(mask_gan)
    mask_gan[mask_gan == 0] = 1
    mask_gan[mask_gan != 1] = 0
#    mask_gan[(mask != 0) * (mask_gan != 1) ] = 2
    plt.figure()
    plt.imshow(mask_gan)
    plt.show(block=False)

    mask_gans_trn = mask_gan * mask_sar
#    mask_gans_trn[mask_gans_trn == 3] = 0
#    mask_gans_trn[mask_gans_trn == 2] = 1
    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    plt.figure()
    plt.imshow(cloud_mask)
    plt.show(block=True)

    sar1 = minmaxnormalization(sar1, mask_gans_trn)
    sar2 = minmaxnormalization(sar2, mask_gans_trn)
    opt_source = minmaxnormalization(np.float32(opt_source), mask_gans_trn)
    opt_target = minmaxnormalization(np.float32(opt_target), mask_gans_trn)
    img_source = np.concatenate((sar1, sar2, opt_source), axis=2)

    extract_patches_stride(ksize,
                           mask=mask_gans_trn,
                           cloud_mask=cloud_mask,
                           output_folder=patch_trn,
                           sar=img_source,
                           opt=opt_target,
                           show=True)


def create_dataset_case_c_multiresolution(ksize=256,
                                          dataset=None,
                                          mask_path=None,
                                          sar_path1=None,
                                          sar_path2=None,
                                          opt_path1=None,
                                          opt_path2=None,
                                          region=None):

    patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/'
    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)
    mask, sar1, opt_target, cloud_mask1 = load_images(mask_path=None,
                                                      sar_path=sar_path1,
                                                      opt_path=opt_path1)
    _, sar2, opt_source, cloud_mask2 = load_images(mask_path=None,
                                                   sar_path=sar_path2,
                                                   opt_path=opt_path2)
    opt_target[np.isnan(opt_target)] = 0
    opt_source[np.isnan(opt_source)] = 0
    cloud_mask = cloud_mask1+cloud_mask2
    cloud_mask[cloud_mask != 0] = 1
    sar1[sar1 > 1.0] = 1.0
    sar2[sar2 > 1.0] = 1.0
    mask_sar = sar1[:, :, 0].copy()
    mask_sar[sar1[:, :, 0] < 1] = 1
    mask_sar[sar1[:, :, 0] == 1] = 0
    mask_gan = np.load('mask_gan_original.npy')
    mask_gan[mask == 0] = 0
    mask_gan[mask_gan != 0] = 2
    mask_gan[(mask != 0) * (mask_gan == 0)] = 1
#    mask_gan = np.load('mask_gan_original.npy')
#    mask_gan[mask_gan == 0] = 1
#    mask_gan[mask_gan != 1] = 0
#    mask_gan[(mask != 0) * (mask_gan != 1) ] = 2
    plt.figure()
    plt.imshow(cloud_mask1)
    plt.show(block=False)

    mask_gans_trn = mask_gan * mask_sar

    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    plt.figure()
    plt.imshow(cloud_mask)
    plt.show(block=False)

    sar1 = minmaxnormalization(sar1, mask_sar)
    sar2 = minmaxnormalization(sar2, mask_sar)
    mask_gans_trn_opt = resampler(mask_gans_trn)
    opt_source = minmaxnormalization(opt_source, mask_gans_trn_opt)
    opt_target = minmaxnormalization(opt_target, mask_gans_trn_opt)
    img_sar = np.concatenate((sar1, sar2), axis=2)
    img_opt = np.concatenate((opt_source, opt_target), axis=2)


    extract_patches_stride_multiresolution(3*ksize,
                                           mask=mask_gans_trn,
                                           mask_sar=cloud_mask,
                                           output_folder=patch_trn,
                                           sar=img_sar,
                                           opt=img_opt,
                                           show=True)


def create_dataset_case_d_multiresolution(ksize=256,
                                          dataset=None,
                                          mask_path=None,
                                          landsat_path=None,
                                          sent2_path=None,
                                          sent1_path=None,
                                          region=None):
    mask_path = 'TrainTestMask_50_50_Dec.tif'
    patch_trn = '/mnt/Data/Pix2Pix_datasets/'+dataset+'/train/'
    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)
    
    img_landsat, cloud_mask = load_landsat(landsat_path)
    img_sent2 = load_sentinel2(sent2_path)
    sar = load_sar(sent1_path)
    mask = load_tiff_image(mask_path)

    img_landsat[np.isnan(img_landsat)] = 0
    img_sent2[np.isnan(img_sent2)] = 0
    cloud_mask[cloud_mask != 0] = 1
    sar[sar > 1.0] = 1.0
    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 0].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0
    mask_gan = np.load('mask_gan_original.npy')
    mask_gan[mask == 0] = 0
    mask_gan[mask_gan != 0] = 2
    mask_gan[(mask != 0) * (mask_gan == 0)] = 1
    mask_gan[mask_gan != 0] = 1

    mask_gans_trn = mask_gan

    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    plt.figure()
    plt.imshow(cloud_mask)
    plt.show(block=False)

    mask_gans_trn_opt = resampler(mask_gans_trn)
    sar = minmaxnormalization(sar, mask_sar)
    img_sent2 = minmaxnormalization(img_sent2, mask_gans_trn)
    img_landsat = minmaxnormalization(img_landsat, mask_gans_trn_opt)
    cloud_mask = scipy.misc.imresize(cloud_mask, mask_sar.shape, interp ='nearest', mode='F')
    extract_patches_stride_multiresolutionx3(ksize,
                                             mask=mask_gans_trn,
                                             mask_sar=mask_sar,
                                             cloud_mask=cloud_mask,
                                             output_folder=patch_trn,
                                             sar=sar,
                                             img_landsat=img_landsat,
                                             img_sent2=img_sent2,
                                             show=True)


def load_data2(SAR, Opt, flip=True, is_test=False):
    img_AB = np.concatenate((SAR, Opt), axis=2)
    # img_AB = preprocess_A_and_B(img_AB, load_size=286, fine_size=256, flip=True, is_test=False)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_data(image_path, flip=True, is_test=False):
    img_AB = np.load(image_path)
    img_AB = preprocess_A_and_B(img_AB, load_size=286, fine_size=256, flip=True, is_test=False)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_data_multiresolution(image_path, flip=True, is_test=False):
    img_A, img_B = np.load(image_path)
    img_A, img_B = preprocess_A_and_B_multiresolution(img_A, img_B, flip=flip, is_test=is_test)


    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_A, img_B


def load_data_multiresolution_D(image_path, flip=True, is_test=False):
    img_A, img_B, img_C = np.load(image_path)
    img_A, img_B, img_C = preprocess_A_and_B_multiresolution_D(img_A, img_B, img_C, flip=flip, is_test=is_test)

    return img_A, img_B, img_C
#    img_AB = np.load(image_path)
#    img_AB = preprocess_A_and_B(img_AB, load_size=286, fine_size=256, flip=True, is_test=False)
#    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
#    return img_AB


def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B


def preprocess_A_and_B_multiresolution(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
#        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
#        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
        print 'Test ...'
    else:
#        img_A = scipy.misc.imresize(img_A, [3*load_size, 3*load_size])
#        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        im_A = np.zeros((3*load_size, 3*load_size, img_A.shape[2]), dtype=img_A.dtype)
        im_B = np.zeros((load_size, load_size, img_B.shape[2]), dtype=img_B.dtype)
        for i in range(img_A.shape[2]):
            im_A[:, :, i] = scipy.misc.imresize(img_A[:, :, i], (3*load_size, 3*load_size), mode='F')
        for i in range(img_B.shape[2]):
            im_B[:, :, i] = scipy.misc.imresize(img_B[:, :, i], (load_size, load_size), mode='F')
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        im_A = im_A[3*h1:3*h1+3*fine_size, 3*w1:3*w1+3*fine_size]
        im_B = im_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            im_A = np.fliplr(im_A)
            im_B = np.fliplr(im_B)
##        elif flip and np.random.random() < 0.1:
##            img_AB = np.flipud(img_AB)

    return im_A, im_B


def preprocess_A_and_B_multiresolution_D(img_A, img_B, img_C, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        print 'Test ...'
    else:
        im_A = np.zeros((load_size, load_size, img_A.shape[2]), dtype=img_A.dtype)
        im_B = np.zeros((load_size, load_size, img_B.shape[2]), dtype=img_B.dtype)
        im_C = np.zeros((load_size//3, load_size//3, img_C.shape[2]), dtype=img_B.dtype)
        for i in range(img_A.shape[2]):
            im_A[:, :, i] = scipy.misc.imresize(img_A[:, :, i], (load_size, load_size), mode='F')
        for i in range(img_B.shape[2]):
            im_B[:, :, i] = scipy.misc.imresize(img_B[:, :, i], (load_size, load_size), mode='F')
        for i in range(img_C.shape[2]):
            im_C[:, :, i] = scipy.misc.imresize(img_C[:, :, i], (load_size//3, load_size//3), mode='F')
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        im_A = im_A[h1:h1+fine_size, w1:w1+fine_size]
        im_B = im_B[h1:h1+fine_size, w1:w1+fine_size]
        im_C = im_C[h1//3:h1//3+fine_size//3, w1//3:w1//3+fine_size//3]

        if flip and np.random.random() > 0.5:
            im_A = np.fliplr(im_A)
            im_B = np.fliplr(im_B)
            im_C = np.fliplr(im_C)
##        elif flip and np.random.random() < 0.1:
##            img_AB = np.flipud(img_AB)

    return img_A, img_B, img_C


def preprocess_A_and_B(img_AB, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
#        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
#        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
        print 'Test ...'
    else:
#        img_AB = scipy.misc.imresize(img_AB, [load_size, load_size])
#        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        im = np.zeros((load_size, load_size, img_AB.shape[2]), dtype=img_AB.dtype)
        for i in range(img_AB.shape[2]):
            im[:, :, i] = scipy.misc.imresize(img_AB[:, :, i], (load_size, load_size), mode='F')
        img_AB = im
#        img_AB = resize(img_AB, (load_size, load_size), preserve_range=True)

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_AB = img_AB[h1:h1+fine_size, w1:w1+fine_size]
#        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        flip_creterion = np.random.random()
        if flip and flip_creterion > 0.9:
            img_AB = np.flipud(img_AB)
#            img_B = np.fliplr(img_B)
        elif flip and flip_creterion > 0.5:
            img_AB = np.fliplr(img_AB)

    return img_AB

def plot_patch(patch, n_fig):
    im =  inverse_transform(patch[:, :, [3, 2, 1]])
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.02)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.02)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.02)
    plt.figure(n_fig)
    plt.imshow(im)
    plt.show(block=False)
    plt.pause(0.5)
    return plt


def plot_samples(self, data_list, idx=6):
    batch_images = load_data_Dic_Multiresolution(samples_list=data_list,
                                                 idxc=idx,
                                                 load_size=self.load_size,
                                                 fine_size=self.fine_size,
                                                 random_transformation=False,
                                                 multitemporal=False)

    sar_t0 = batch_images[0].reshape(self.batch_size, 3*self.image_size, 3*self.image_size, -1)
    opt_t0 = batch_images[1]        
    opt_t0_fake = self.sess.run([self.fake_B],
                                feed_dict={self.sar: sar_t0, self.dropout_d: False, self.dropout_g: True})
    print(np.shape(opt_t0_fake))
    plot_patch(opt_t0, n_fig='real opt')
    plot_patch(np.array(opt_t0_fake).reshape(self.image_size, self.image_size, -1), n_fig='fake opt')
    return 0


def save_samples(self, data_list, output_path, idx=6, epoch=0, multitemporal=False):
    batch_images = load_data_Dic_Multiresolution(samples_list=data_list,
                                                 idxc=idx,
                                                 load_size=self.load_size,
                                                 fine_size=self.fine_size,
                                                 random_transformation=False,
                                                 multitemporal=multitemporal)
    if multitemporal:
        sar_t0 = batch_images[0].reshape(self.batch_size, self.image_size, self.image_size, -1)
        opt_t0 = batch_images[1]
        sar_t1 = batch_images[2].reshape(self.batch_size, self.image_size, self.image_size, -1)
        opt_t1 = batch_images[3].reshape(self.batch_size, self.image_size, self.image_size, -1)
        opt_t0_fake = self.sess.run([self.fake_B_sample],
                                    feed_dict={self.sar_t0: sar_t0, self.sar_t1: sar_t1, self.opt_t1: opt_t1})
        opt_t0_fake = np.array(opt_t0_fake).reshape(self.image_size, self.image_size, -1)
        opt_t0_fake =  inverse_transform(opt_t0_fake[:, :, [3, 2, 1]])
        opt_t0_fake[:, :, 0] = exposure.equalize_adapthist(opt_t0_fake[:, :, 0], clip_limit=0.02)
        opt_t0_fake[:, :, 1] = exposure.equalize_adapthist(opt_t0_fake[:, :, 1], clip_limit=0.02)
        opt_t0_fake[:, :, 2] = exposure.equalize_adapthist(opt_t0_fake[:, :, 2], clip_limit=0.02)

        opt_t0 =  inverse_transform(opt_t0[:, :, [3, 2, 1]])
        opt_t0[:, :, 0] = exposure.equalize_adapthist(opt_t0[:, :, 0], clip_limit=0.02)
        opt_t0[:, :, 1] = exposure.equalize_adapthist(opt_t0[:, :, 1], clip_limit=0.02)
        opt_t0[:, :, 2] = exposure.equalize_adapthist(opt_t0[:, :, 2], clip_limit=0.02)

        scipy.misc.imsave(output_path + '/opt_real.tiff', opt_t0)
        scipy.misc.imsave(output_path + '/opt_fake_' + str(epoch) + '.tiff', opt_t0_fake)
    else:
        sar_t0 = batch_images[0].reshape(self.batch_size, self.image_size, self.image_size, -1)
        opt_t0 = batch_images[1]        
        opt_t0_fake = self.sess.run([self.fake_B_sample],
                                    feed_dict={self.sar: sar_t0})
        opt_t0_fake = np.array(opt_t0_fake).reshape(self.image_size, self.image_size, -1)
        opt_t0_fake =  inverse_transform(opt_t0_fake[:, :, [3, 2, 1]])
        opt_t0_fake[:, :, 0] = exposure.equalize_adapthist(opt_t0_fake[:, :, 0], clip_limit=0.02)
        opt_t0_fake[:, :, 1] = exposure.equalize_adapthist(opt_t0_fake[:, :, 1], clip_limit=0.02)
        opt_t0_fake[:, :, 2] = exposure.equalize_adapthist(opt_t0_fake[:, :, 2], clip_limit=0.02)

        opt_t0 =  inverse_transform(opt_t0[:, :, [3, 2, 1]])
        opt_t0[:, :, 0] = exposure.equalize_adapthist(opt_t0[:, :, 0], clip_limit=0.02)
        opt_t0[:, :, 1] = exposure.equalize_adapthist(opt_t0[:, :, 1], clip_limit=0.02)
        opt_t0[:, :, 2] = exposure.equalize_adapthist(opt_t0[:, :, 2], clip_limit=0.02)

        scipy.misc.imsave(output_path + '/opt_real.tiff', opt_t0)
        scipy.misc.imsave(output_path + '/opt_fake_' + str(epoch) + '.tiff', opt_t0_fake)

    return 0


def save_samples_multiresolution(self, data_list, output_path, idx=6, epoch=0, multitemporal=False):
    batch_images = load_data_Dic_Multiresolution(samples_list=data_list,
                                                 idxc=idx,
                                                 load_size=self.load_size,
                                                 fine_size=self.fine_size,
                                                 random_transformation=False,
                                                 multitemporal=multitemporal)
    if multitemporal:
        sar_t0 = batch_images[0].reshape(self.batch_size, 3*self.image_size, 3*self.image_size, -1)
        opt_t0 = batch_images[1]
        sar_t1 = batch_images[2].reshape(self.batch_size, 3*self.image_size, 3*self.image_size, -1)
        opt_t1 = batch_images[3].reshape(self.batch_size, self.image_size, self.image_size, -1)
        opt_t0_fake = self.sess.run([self.fake_B_sample],
                                    feed_dict={self.sar_t0: sar_t0, self.sar_t1: sar_t1, self.opt_t1: opt_t1})
        opt_t0_fake = np.array(opt_t0_fake).reshape(self.image_size, self.image_size, -1)
        opt_t0_fake =  inverse_transform(opt_t0_fake[:, :, [3, 2, 1]])
        opt_t0_fake[:, :, 0] = exposure.equalize_adapthist(opt_t0_fake[:, :, 0], clip_limit=0.02)
        opt_t0_fake[:, :, 1] = exposure.equalize_adapthist(opt_t0_fake[:, :, 1], clip_limit=0.02)
        opt_t0_fake[:, :, 2] = exposure.equalize_adapthist(opt_t0_fake[:, :, 2], clip_limit=0.02)

        opt_t0 =  inverse_transform(opt_t0[:, :, [3, 2, 1]])
        opt_t0[:, :, 0] = exposure.equalize_adapthist(opt_t0[:, :, 0], clip_limit=0.02)
        opt_t0[:, :, 1] = exposure.equalize_adapthist(opt_t0[:, :, 1], clip_limit=0.02)
        opt_t0[:, :, 2] = exposure.equalize_adapthist(opt_t0[:, :, 2], clip_limit=0.02)

        scipy.misc.imsave(output_path + '/opt_real.tiff', opt_t0)
        scipy.misc.imsave(output_path + '/opt_fake_' + str(epoch) + '.tiff', opt_t0_fake)
    else:
        sar_t0 = batch_images[0].reshape(self.batch_size, 3*self.image_size, 3*self.image_size, -1)
        opt_t0 = batch_images[1]        
        opt_t0_fake = self.sess.run([self.fake_B_sample],
                                    feed_dict={self.sar: sar_t0})
        opt_t0_fake = np.array(opt_t0_fake).reshape(self.image_size, self.image_size, -1)
        opt_t0_fake =  inverse_transform(opt_t0_fake[:, :, [3, 2, 1]])
        opt_t0_fake[:, :, 0] = exposure.equalize_adapthist(opt_t0_fake[:, :, 0], clip_limit=0.02)
        opt_t0_fake[:, :, 1] = exposure.equalize_adapthist(opt_t0_fake[:, :, 1], clip_limit=0.02)
        opt_t0_fake[:, :, 2] = exposure.equalize_adapthist(opt_t0_fake[:, :, 2], clip_limit=0.02)

        opt_t0 =  inverse_transform(opt_t0[:, :, [3, 2, 1]])
        opt_t0[:, :, 0] = exposure.equalize_adapthist(opt_t0[:, :, 0], clip_limit=0.02)
        opt_t0[:, :, 1] = exposure.equalize_adapthist(opt_t0[:, :, 1], clip_limit=0.02)
        opt_t0[:, :, 2] = exposure.equalize_adapthist(opt_t0[:, :, 2], clip_limit=0.02)

        scipy.misc.imsave(output_path + '/opt_real.tiff', opt_t0)
        scipy.misc.imsave(output_path + '/opt_fake_' + str(epoch) + '.tiff', opt_t0_fake)

    return 0

# def tranformations(sar_t0,
#                    opt_t0,
#                    sar_t1=None,
#                    opt_t1=None,
#                    load_size=None,
#                    fine_size=None,
#                    random_transformation=False,
#                    num_transformations=2,
#                    labels=False):
    
#     h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
#     w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))

#     trans_creterion = np.random.random()
#     transformation = np.random.randint(num_transformations) # tranformation type
#     if (sar_t1 is None) or (opt_t1 is None):
#         sar_t0 = np.float32(resize(sar_t0, (load_size, load_size), preserve_range=True))
#         opt_t0 = np.float32(resize(opt_t0, (load_size, load_size), preserve_range=True))
#         sar_t0 = sar_t0[h1:h1+fine_size, w1:w1+fine_size]
#         opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
#         if labels is not False:
#             labels = resize(labels, (load_size, load_size), order=0, preserve_range=True)
#             labels = labels[h1:h1+fine_size, w1:w1+fine_size]

#         if random_transformation and trans_creterion > 0.5:
#             if transformation == 0:
#                 sar_t0 = np.flipud(sar_t0)
#                 opt_t0 = np.flipud(opt_t0)
#                 if labels is not False:
#                     labels = np.flipud(labels)
#             if transformation == 1:
#                 sar_t0 = np.fliplr(sar_t0)
#                 opt_t0 = np.fliplr(opt_t0)
#                 if labels is not False:
#                     labels = np.fliplr(labels)
#             if transformation == 2:
#                 sar_t0 = np.rot90(sar_t0, 3, (0, 1))
#                 opt_t0 = np.rot90(opt_t0, 3, (0, 1))
#                 if labels is not False:
#                     labels = np.rot90(labels, 3, (0, 1))
#             if transformation == 3:
#                 sar_t0 = np.rot90(sar_t0, 2, (0, 1))
#                 opt_t0 = np.rot90(opt_t0, 2, (0, 1))
#                 if labels is not False:
#                     labels = np.rot90(labels, 2, (0, 1))
#             if transformation == 4:
#                 sar_t0 = np.rot90(sar_t0, 1, (0, 1))
#                 opt_t0 = np.rot90(opt_t0, 1, (0, 1))
#                 if labels is not False:
#                     labels = np.rot90(labels, 1, (0, 1))
#         return sar_t0, opt_t0, labels
#     else:
#         sar_t0 = np.float32(resize(sar_t0, (load_size, load_size), preserve_range=True))
#         opt_t0 = np.float32(resize(opt_t0, (load_size, load_size), preserve_range=True))
#         sar_t1 = np.float32(resize(sar_t1, (load_size, load_size), preserve_range=True))
#         opt_t1 = np.float32(resize(opt_t1, (load_size, load_size), preserve_range=True))
#         sar_t0 = sar_t0[h1:h1+fine_size, w1:w1+fine_size]
#         opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
#         sar_t1 = sar_t1[h1:h1+fine_size, w1:w1+fine_size]
#         opt_t1 = opt_t1[h1:h1+fine_size, w1:w1+fine_size]
#         if labels is not False:
#             labels = resize(labels, (load_size, load_size), order=0, preserve_range=True)
#             labels = labels[h1:h1+fine_size, w1:w1+fine_size]
        
#         if random_transformation and trans_creterion > 0.5:
#             if transformation == 0:
#                 sar_t0 = np.flipud(sar_t0)
#                 opt_t0 = np.flipud(opt_t0)
#                 sar_t1 = np.flipud(sar_t1)
#                 opt_t1 = np.flipud(opt_t1)
#                 if labels is not False:
#                     labels = np.flipud(labels)
#             if transformation == 1:
#                 sar_t0 = np.fliplr(sar_t0)
#                 opt_t0 = np.fliplr(opt_t0)
#                 sar_t1 = np.fliplr(sar_t1)
#                 opt_t1 = np.fliplr(opt_t1)
#                 if labels is not False:
#                     labels = np.fliplr(labels)
#             if transformation == 2:
#                 sar_t0 = np.rot90(sar_t0, 3, (0, 1))
#                 opt_t0 = np.rot90(opt_t0, 3, (0, 1))
#                 sar_t1 = np.rot90(sar_t1, 3, (0, 1))
#                 opt_t1 = np.rot90(opt_t1, 3, (0, 1))
#                 if labels is not False:
#                     labels = np.rot90(labels, 3, (0, 1))
#             if transformation == 3:
#                 sar_t0 = np.rot90(sar_t0, 2, (0, 1))
#                 opt_t0 = np.rot90(opt_t0, 2, (0, 1))
#                 sar_t1 = np.rot90(sar_t1, 2, (0, 1))
#                 opt_t1 = np.rot90(opt_t1, 2, (0, 1))
#                 if labels is not False:
#                     labels = np.rot90(labels, 2, (0, 1))
#             if transformation == 4:
#                 sar_t0 = np.rot90(sar_t0, 1, (0, 1))
#                 opt_t0 = np.rot90(opt_t0, 1, (0, 1))
#                 sar_t1 = np.rot90(sar_t1, 1, (0, 1))
#                 opt_t1 = np.rot90(opt_t1, 1, (0, 1))
#                 if labels is not False:
#                     labels = np.rot90(labels, 1, (0, 1))

#         return sar_t0, opt_t0, sar_t1, opt_t1, labels


def tranformations(sar_t0,
                   opt_t0,
                   sar_t1=None,
                   opt_t1=None,
                   load_size=None,
                   fine_size=None,
                   random_transformation=False,
                   num_transformations=2,
                   labels=False):
    
    h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))

    trans_creterion = np.random.random()
    transformation = np.random.randint(num_transformations) # tranformation type
    if (sar_t1 is None) or (opt_t1 is None):
        sar_t0 = np.float32(resize(sar_t0, (load_size, load_size), preserve_range=True))
        opt_t0 = np.float32(resize(opt_t0, (load_size, load_size), preserve_range=True))
        sar_t0 = sar_t0[h1:h1+fine_size, w1:w1+fine_size]
        opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
        if labels is not False:
            labels = resize(labels, (load_size, load_size), order=0, preserve_range=True)
            labels = labels[h1:h1+fine_size, w1:w1+fine_size]
        trans_creterion = np.random.random()
        if random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 3, (0, 1))
            opt_t0 = np.rot90(opt_t0, 3, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 3, (0, 1))
        elif random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 2, (0, 1))
            opt_t0 = np.rot90(opt_t0, 2, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 2, (0, 1))
        elif random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 1, (0, 1))
            opt_t0 = np.rot90(opt_t0, 1, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 1, (0, 1))
        elif random_transformation and trans_creterion > 0.75:
            sar_t0 = np.flipud(sar_t0)
            opt_t0 = np.flipud(opt_t0)
            if labels is not False:
                labels = np.flipud(labels)
        elif random_transformation and trans_creterion > 0.50:
            sar_t0 = np.fliplr(sar_t0)
            opt_t0 = np.fliplr(opt_t0)
            if labels is not False:
                labels = np.fliplr(labels)
        return sar_t0, opt_t0, labels
    else:
        sar_t0 = np.float32(resize(sar_t0, (load_size, load_size), preserve_range=True))
        opt_t0 = np.float32(resize(opt_t0, (load_size, load_size), preserve_range=True))
        sar_t1 = np.float32(resize(sar_t1, (load_size, load_size), preserve_range=True))
        opt_t1 = np.float32(resize(opt_t1, (load_size, load_size), preserve_range=True))
        sar_t0 = sar_t0[h1:h1+fine_size, w1:w1+fine_size]
        opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
        sar_t1 = sar_t1[h1:h1+fine_size, w1:w1+fine_size]
        opt_t1 = opt_t1[h1:h1+fine_size, w1:w1+fine_size]
        if labels is not False:
            labels = resize(labels, (load_size, load_size), order=0, preserve_range=True)
            labels = labels[h1:h1+fine_size, w1:w1+fine_size]
        trans_creterion = np.random.random()
        if random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 3, (0, 1))
            opt_t0 = np.rot90(opt_t0, 3, (0, 1))
            sar_t1 = np.rot90(sar_t1, 3, (0, 1))
            opt_t1 = np.rot90(opt_t1, 3, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 3, (0, 1))
        elif random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 2, (0, 1))
            opt_t0 = np.rot90(opt_t0, 2, (0, 1))
            sar_t1 = np.rot90(sar_t1, 2, (0, 1))
            opt_t1 = np.rot90(opt_t1, 2, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 2, (0, 1))
        elif random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 1, (0, 1))
            opt_t0 = np.rot90(opt_t0, 1, (0, 1))
            sar_t1 = np.rot90(sar_t1, 1, (0, 1))
            opt_t1 = np.rot90(opt_t1, 1, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 1, (0, 1))
        elif random_transformation and trans_creterion > 0.75:
            sar_t0 = np.flipud(sar_t0)
            opt_t0 = np.flipud(opt_t0)
            sar_t1 = np.flipud(sar_t1)
            opt_t1 = np.flipud(opt_t1)
            if labels is not False:
                labels = np.flipud(labels)
        elif random_transformation and trans_creterion > 0.50:
            sar_t0 = np.fliplr(sar_t0)
            opt_t0 = np.fliplr(opt_t0)
            sar_t1 = np.fliplr(sar_t1)
            opt_t1 = np.fliplr(opt_t1)
            if labels is not False:
                labels = np.fliplr(labels)
        return sar_t0, opt_t0, sar_t1, opt_t1, labels


def transformations_multiresolution(sar_t0,
                                    opt_t0,
                                    sar_t1=None,
                                    opt_t1=None,
                                    load_size=286,
                                    fine_size=256,
                                    random_transformation=False,
                                    labels=False):
    # TODO: create a random discrete vairable to select the transformation
    h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    if (sar_t1 is None) or (opt_t1 is None):
        sar_t0 = np.float32(resize(sar_t0, (3*load_size, 3*load_size), preserve_range=True))
        opt_t0 = np.float32(resize(opt_t0, (load_size, load_size), preserve_range=True))
        sar_t0 = sar_t0[3*h1:3*h1+3*fine_size, 3*w1:3*w1+3*fine_size]
        opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
        if labels is not False:
            labels = resize(labels, (load_size, load_size), order=0, preserve_range=True)
            labels = labels[h1:h1+fine_size, w1:w1+fine_size]
        trans_creterion = np.random.random()
        if random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 3, (0, 1))
            opt_t0 = np.rot90(opt_t0, 3, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 3, (0, 1))
        elif random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 2, (0, 1))
            opt_t0 = np.rot90(opt_t0, 2, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 2, (0, 1))
        elif random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 1, (0, 1))
            opt_t0 = np.rot90(opt_t0, 1, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 1, (0, 1))
        elif random_transformation and trans_creterion > 0.75:
            sar_t0 = np.flipud(sar_t0)
            opt_t0 = np.flipud(opt_t0)
            if labels is not False:
                labels = np.flipud(labels)
        elif random_transformation and trans_creterion > 0.50:
            sar_t0 = np.fliplr(sar_t0)
            opt_t0 = np.fliplr(opt_t0)
            if labels is not False:
                labels = np.fliplr(labels)
        return sar_t0, opt_t0, labels
    else:
        sar_t0 = np.float32(resize(sar_t0, (3*load_size, 3*load_size), preserve_range=True))
        opt_t0 = np.float32(resize(opt_t0, (load_size, load_size), preserve_range=True))
        sar_t1 = np.float32(resize(sar_t1, (3*load_size, 3*load_size), preserve_range=True))
        opt_t1 = np.float32(resize(opt_t1, (load_size, load_size), preserve_range=True))
        sar_t0 = sar_t0[3*h1:3*h1+3*fine_size, 3*w1:3*w1+3*fine_size]
        opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
        sar_t1 = sar_t1[3*h1:3*h1+3*fine_size, 3*w1:3*w1+3*fine_size]
        opt_t1 = opt_t1[h1:h1+fine_size, w1:w1+fine_size]
        if labels is not False:
            labels = resize(labels, (load_size, load_size), order=0, preserve_range=True)
            labels = labels[h1:h1+fine_size, w1:w1+fine_size]
        trans_creterion = np.random.random()
        if random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 3, (0, 1))
            opt_t0 = np.rot90(opt_t0, 3, (0, 1))
            sar_t1 = np.rot90(sar_t1, 3, (0, 1))
            opt_t1 = np.rot90(opt_t1, 3, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 3, (0, 1))
        elif random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 2, (0, 1))
            opt_t0 = np.rot90(opt_t0, 2, (0, 1))
            sar_t1 = np.rot90(sar_t1, 2, (0, 1))
            opt_t1 = np.rot90(opt_t1, 2, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 2, (0, 1))
        elif random_transformation and trans_creterion > 1:
            sar_t0 = np.rot90(sar_t0, 1, (0, 1))
            opt_t0 = np.rot90(opt_t0, 1, (0, 1))
            sar_t1 = np.rot90(sar_t1, 1, (0, 1))
            opt_t1 = np.rot90(opt_t1, 1, (0, 1))
            if labels is not False:
                labels = np.rot90(labels, 1, (0, 1))
        elif random_transformation and trans_creterion > 0.75:
            sar_t0 = np.flipud(sar_t0)
            opt_t0 = np.flipud(opt_t0)
            sar_t1 = np.flipud(sar_t1)
            opt_t1 = np.flipud(opt_t1)
            if labels is not False:
                labels = np.flipud(labels)
        elif random_transformation and trans_creterion > 0.50:
            sar_t0 = np.fliplr(sar_t0)
            opt_t0 = np.fliplr(opt_t0)
            sar_t1 = np.fliplr(sar_t1)
            opt_t1 = np.fliplr(opt_t1)
            if labels is not False:
                labels = np.fliplr(labels)
        return sar_t0, opt_t0, sar_t1, opt_t1, labels


def transformations_multiresolution2(sar_t0,
                                     opt_t0,
                                     sar_t1=None,
                                     opt_t1=None,
                                     load_size=286,
                                     fine_size=256,
                                     random_transformation=False,
                                     num_transformations=2,
                                     labels=False):
    # TODO: create a random discrete vairable to select the transformation
    trans_creterion = np.random.random()
    transformation = np.random.randint(num_transformations)

    h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    if (sar_t1 is None) or (opt_t1 is None):
        sar_t0 = np.float32(resize(sar_t0, (3*load_size, 3*load_size), preserve_range=True))
        opt_t0 = np.float32(resize(opt_t0, (load_size, load_size), preserve_range=True))
        sar_t0 = sar_t0[3*h1:3*h1+3*fine_size, 3*w1:3*w1+3*fine_size]
        opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
        if labels is not False:
            labels = resize(labels, (load_size, load_size), order=0, preserve_range=True)
            labels = labels[h1:h1+fine_size, w1:w1+fine_size]

        if random_transformation and trans_creterion > 0.5:
            if transformation==0:
                sar_t0 = np.flipud(sar_t0)
                opt_t0 = np.flipud(opt_t0)
                if labels is not False:
                    labels = np.flipud(labels)
            if transformation==1:
                sar_t0 = np.fliplr(sar_t0)
                opt_t0 = np.fliplr(opt_t0)
                if labels is not False:
                    labels = np.fliplr(labels)
            if transformation==2:
                sar_t0 = np.rot90(sar_t0, 3, (0, 1))
                opt_t0 = np.rot90(opt_t0, 3, (0, 1))
                if labels is not False:
                    labels = np.rot90(labels, 3, (0, 1))
            if transformation==3:
                sar_t0 = np.rot90(sar_t0, 2, (0, 1))
                opt_t0 = np.rot90(opt_t0, 2, (0, 1))
                if labels is not False:
                    labels = np.rot90(labels, 2, (0, 1))
            if transformation==4:
                sar_t0 = np.rot90(sar_t0, 1, (0, 1))
                opt_t0 = np.rot90(opt_t0, 1, (0, 1))
                if labels is not False:
                    labels = np.rot90(labels, 1, (0, 1))
            # more transformations
        return sar_t0, opt_t0, labels
    else:
        sar_t0 = np.float32(resize(sar_t0, (3*load_size, 3*load_size), preserve_range=True))
        opt_t0 = np.float32(resize(opt_t0, (load_size, load_size), preserve_range=True))
        sar_t1 = np.float32(resize(sar_t1, (3*load_size, 3*load_size), preserve_range=True))
        opt_t1 = np.float32(resize(opt_t1, (load_size, load_size), preserve_range=True))
        sar_t0 = sar_t0[3*h1:3*h1+3*fine_size, 3*w1:3*w1+3*fine_size]
        opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
        sar_t1 = sar_t1[3*h1:3*h1+3*fine_size, 3*w1:3*w1+3*fine_size]
        opt_t1 = opt_t1[h1:h1+fine_size, w1:w1+fine_size]
        if labels is not False:
            labels = resize(labels, (load_size, load_size), order=0, preserve_range=True)
            labels = labels[h1:h1+fine_size, w1:w1+fine_size]
        
        if random_transformation and trans_creterion > 0.5:
            if transformation==0:
                sar_t0 = np.flipud(sar_t0)
                opt_t0 = np.flipud(opt_t0)
                sar_t1 = np.flipud(sar_t1)
                opt_t1 = np.flipud(opt_t1)
                if labels is not False:
                    labels = np.flipud(labels)
            if transformation==1:
                sar_t0 = np.fliplr(sar_t0)
                opt_t0 = np.fliplr(opt_t0)
                sar_t1 = np.fliplr(sar_t1)
                opt_t1 = np.fliplr(opt_t1)
                if labels is not False:
                    labels = np.fliplr(labels)
            if transformation==2:
                sar_t0 = np.rot90(sar_t0, 3, (0, 1))
                opt_t0 = np.rot90(opt_t0, 3, (0, 1))
                sar_t1 = np.rot90(sar_t1, 3, (0, 1))
                opt_t1 = np.rot90(opt_t1, 3, (0, 1))
                if labels is not False:
                    labels = np.rot90(labels, 3, (0, 1))
            if transformation==3:
                sar_t0 = np.rot90(sar_t0, 2, (0, 1))
                opt_t0 = np.rot90(opt_t0, 2, (0, 1))
                sar_t1 = np.rot90(sar_t1, 2, (0, 1))
                opt_t1 = np.rot90(opt_t1, 2, (0, 1))
                if labels is not False:
                    labels = np.rot90(labels, 2, (0, 1))
            if transformation==4:
                sar_t0 = np.rot90(sar_t0, 1, (0, 1))
                opt_t0 = np.rot90(opt_t0, 1, (0, 1))
                sar_t1 = np.rot90(sar_t1, 1, (0, 1))
                opt_t1 = np.rot90(opt_t1, 1, (0, 1))
                if labels is not False:
                    labels = np.rot90(labels, 1, (0, 1))
            # more transformations
        return sar_t0, opt_t0, sar_t1, opt_t1, labels


def preprocess_S_and_O_and_L(img_AB, labels, load_size=286, fine_size=256, flip=True, is_test=False):

    img_AB = np.float32(resize(img_AB, (load_size, load_size), preserve_range=True))
    labels = resize(np.uint8(labels), (load_size, load_size), order=0, preserve_range=True)
    h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    img_AB = img_AB[h1:h1+fine_size, w1:w1+fine_size]
    labels = labels[h1:h1+fine_size, w1:w1+fine_size]
    # plt.figure()
    # plt.imshow(labels)
    # plt.show(block=False)
    trans_creterion = np.random.random()
    if flip and trans_creterion > 0.90:
        img_AB = np.rot90(img_AB, 3, (0, 1))
        labels = np.rot90(labels, 3, (0, 1))
    elif flip and trans_creterion > 0.85:
        img_AB = np.rot90(img_AB, 2, (0, 1))
        labels = np.rot90(labels, 2, (0, 1))
    elif flip and trans_creterion > 0.80:
        img_AB = np.rot90(img_AB, 1, (0, 1))
        labels = np.rot90(labels, 1, (0, 1))
    elif flip and trans_creterion > 0.75:
        img_AB = np.flipud(img_AB)
        labels = np.flipud(labels)
    elif flip and trans_creterion > 0.50:
        img_AB = np.fliplr(img_AB)
        labels = np.fliplr(labels)

    return img_AB, labels


def preprocess_S_and_O_and_L_cropping(img_AB, labels, load_size=286, fine_size=256, flip=True, is_test=False):

    h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    img_AB = img_AB[h1:h1+fine_size, w1:w1+fine_size]
    labels = labels[h1:h1+fine_size, w1:w1+fine_size]
    # plt.figure()
    # plt.imshow(labels)
    # plt.show(block=False)
    trans_creterion = np.random.random()
    if flip and trans_creterion > 0.90:
        img_AB = np.rot90(img_AB, 3, (0, 1))
        labels = np.rot90(labels, 3, (0, 1))
    elif flip and trans_creterion > 0.85:
        img_AB = np.rot90(img_AB, 2, (0, 1))
        labels = np.rot90(labels, 2, (0, 1))
    elif flip and trans_creterion > 0.80:
        img_AB = np.rot90(img_AB, 1, (0, 1))
        labels = np.rot90(labels, 1, (0, 1))
    elif flip and trans_creterion > 0.75:
        img_AB = np.flipud(img_AB)
        labels = np.flipud(labels)
    elif flip and trans_creterion > 0.50:
        img_AB = np.fliplr(img_AB)
        labels = np.fliplr(labels)

    return img_AB, labels

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def compute_metrics(reference, predictions):
    accuracy = accuracy_score(reference, predictions)
    f1 = 100 * f1_score(reference, predictions, average=None)
    rs = 100 * recall_score(reference, predictions, average=None)
    ps = 100 * precision_score(reference, predictions, average=None)
    print ('F1 score')
    print (np.around(f1, decimals=1))
    print ('Recall')
    print (np.around(rs, decimals=1))
    print ('Precision')
    print (np.around(ps, decimals=1))
    print ('accuracy ->', 100 * accuracy)
    print ('Recall ->', rs.mean())
    print ('Precision ->', ps.mean())
    print ('F1 score ->', f1.mean())