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
from time import gmtime, strftime
from osgeo import gdal
import glob
from skimage.transform import resize
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
import keras
import collections

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def balance_data(data, labels, indexs=None, samples_per_class=32):

    if (indexs is None):
        idxs = np.arange(len(labels))
    else:
        idxs = indexs

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
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
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


def resampler(img):
    img_shape = img.shape
    if img.ndim == 3:
        im_c = np.zeros((img_shape[0]+1, img_shape[1], img_shape[2]), dtype=img.dtype)
        im_c[0:-1] = img
        im = np.zeros((int(im_c.shape[0]/3), int(im_c.shape[1]/3), img_shape[2]), dtype=img.dtype)
        for i in range(img.shape[2]):
            im[:, :, i] = scipy.misc.imresize(im_c[:, :, i], (im.shape[0], im.shape[1]), mode='F')
#        im = resize(im, (im.shape[0]/3, im.shape[1]/3), preserve_range=True)
    else:
        im = np.zeros((img_shape[0]+1, img_shape[1]), dtype=img.dtype)
        im[0:-1] = img
        im = scipy.misc.imresize(im, (int(im.shape[0]/3), int(im.shape[1]/3)), interp ='nearest', mode='F')
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


def get_pixel_index(ksize, mask, num_patches=400, show=False):
    rows, cols = mask.shape
    r, _ = np.where(mask == 1)
    top_row = r[0]
    m = mask.copy()
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
#    # Split in training and testing
#    row_trn = row[mask == 1]
#    col_trn = col[mask == 1]
#    row_tst = row[mask == 2]
#    col_tst = col[mask == 2]
    idx = np.arange(0, len(row))
    idx = np.random.choice(idx, num_patches, replace=False)
    row_trn = row[idx]
    col_trn = col[idx]
    idx = np.arange(0, len(row))
    idx = np.random.choice(idx, num_patches, replace=False)
    row_tst = row[idx]
    col_tst = col[idx]
    if show:
        m[row_trn, col_trn] = 5
        plt.figure()
        plt.imshow(m)
        plt.show()

    return row_trn, col_trn, row_tst, col_tst

# idxc = np.random.randint(0, len(samples_list))
#     data_Dic = np.load(samples_list[idxc]).item()
#     labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
#     sar_patch = np.array(data_Dic['sar']).astype('float32')
#     opt_patch = np.array(data_Dic['opt']).astype('float32').reshape(len(labels_patch), self.n_features)
#     labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()

#     trn_samples = opt_patch
#     trn_labels = labels_patch

#     idx_samples = np.arange(0, len(trn_labels))
#     idx_samples = idx_samples[trn_labels == 1]
#     trn_samples = trn_samples[trn_labels == 1]

#     return sar_patch, trn_samples, idx_samples


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


def load_data4Validation_quemadas(self, samples_list):
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
    # trn_samples, trn_labels, _ = balance_data(val_Data, val_Labels, samples_per_class=n_samples_quemadas)
    return val_Data, val_Labels.reshape(len(val_Labels), 1)


def load_data4Classifier(self, samples_list, labels2new_labels):
    idxc = np.random.randint(0, len(samples_list))
    data_Dic = np.load(samples_list[idxc]).item()
    labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
    sar_patch = np.array(data_Dic['sar']).astype('float32')
    opt_patch = np.array(data_Dic['opt']).astype('float32').reshape(len(labels_patch), self.n_features)
    labels_patch = np.array(data_Dic['labels']).astype('float32').ravel()
    index = np.arange(0, len(labels_patch))
    # remove background
    # print (samples_list[idxc])
    # print (np.unique(labels_patch))
    index = index[labels_patch != 0]
    trn_samples = opt_patch[labels_patch != 0]
    trn_labels = labels_patch[labels_patch != 0]
    index = index[trn_labels != 5]
    trn_samples = trn_samples[trn_labels != 5]
    trn_labels = trn_labels[trn_labels != 5]
    index = index[trn_labels != 10]
    trn_samples = trn_samples[trn_labels != 10]
    trn_labels = trn_labels[trn_labels != 10]
    # print (np.unique(trn_labels))
    # Balance data
    trn_samples, trn_labels, idx_samples = balance_data(trn_samples, trn_labels, index=index, samples_per_class=32)
    # convert labels index and also one-hot encoding.
    trn_labels_one_hot = np.zeros((len(trn_labels), self.n_classes), dtype='float32')
    for idx_l in range(len(trn_labels)):
        one_hot = labels2new_labels[trn_labels[idx_l]]
        trn_labels_one_hot[idx_l, one_hot] = 1
    return sar_patch, trn_samples, trn_labels_one_hot, labels_patch, idx_samples


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


def create_training_samples_Classifier(image_path, labels_path, mask_path):
    img, _ = load_landsat(image_path)
    mask = load_tiff_image(mask_path)
    mask = resampler(mask)
    labels = load_tiff_image(labels_path)
    labels = resampler(labels)
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

    # convert class vectors to binary class matrices
    # one_hot_labels = keras.utils.to_categorical(new_labels, num_classes)
    # np.save('trn_samples_classifier', img)
    # np.save('trn_labels_classifier', one_hot_labels)


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
                # np.save(output_folder + str(cont), patch_par)
                cont += 1
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
    m2[mask==2] = 0
    plt.imshow(m2)
    plt.show()
    train_area = m2[m2!=0]
    print(train_area.shape)
    print ()
    return 0


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

    print (output_folder)
    rows, cols = mask.shape
    cont = 0
    m2 = mask.copy()
    for row in range(0, rows - ksize - 1, stride):
        for col in range(0, cols - ksize - 1, stride):
            auxmask = mask[row:row + ksize, col:col + ksize]
            if (np.sum(mask_sar[row:row + ksize, col:col + ksize]) > (ksize**2) / 1.1) and (np.sum(auxmask==0) < (ksize**2) / 1.2) and (np.sum(auxmask==2) == 0):
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
                # np.save(output_folder + str(cont), patch_tuple)
                # Load read_dictionary = np.load('my_file.npy').item()
                cont += 1
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
    # m2[mask==2] = 0
    plt.figure()
    plt.imshow(m2)
    plt.show()
    # train_area = m2[m2!=0]
    # print(train_area.shape)
    # print ()
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
                # print ("Patch valido ...", cont)
                # patch_tuple = {}
                patch_sar = sar[row:row + ksize, col:col + ksize, :]
                # patch_tuple["sar"] = patch_sar
                patch_opt = opt[row:row + ksize, col:col + ksize, :]
                # patch_tuple["opt"] = patch_opt
                # patch_labels = labels[row:row + ksize, col:col + ksize]
                # patch_tuple["labels"] = patch_labels
                patch_tuple = np.concatenate((patch_sar, patch_opt), axis=2)
                # np.save(output_folder + str(cont), patch_tuple)
                # Load read_dictionary = np.load('my_file.npy').item()
                m2[row:row + ksize, col:col + ksize] = np.random.randint(255)
                if (np.sum(mask_val[row:row + ksize, col:col + ksize]) == 256**2):
                    print ("Validation sample -->", cont)
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
                patch_opt = opt[row//3:row//3+ksize//3, col//3:col//+ksize//3, :]
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


#def minmaxnormalization(img, region='all'):
#
#    num_rows, num_cols, bands = img.shape
#    img = img.reshape(num_rows * num_cols, bands)
#    mask_gans_trn = mask_4_trn_gans(num_rows, num_cols, region)
#    scaler = pre.MinMaxScaler((-1, 1)).fit(img[mask_gans_trn.ravel() == 1])
#    img = np.float32(scaler.transform(img))
#    img = img.reshape(num_rows, num_cols, bands)
#
#    return img
def minmaxnormalization(img, mask):

    num_rows, num_cols, bands = img.shape
    img = img.reshape(num_rows * num_cols, bands)
    # print img.shape
    scaler = pre.MinMaxScaler((-1, 1)).fit(img[mask.ravel() == 1])
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
    # plt.close('all')

    sar = minmaxnormalization(sar, mask_sar)
    opt = minmaxnormalization(opt, mask_opt)

    extract_patches_4_classifierQuemadas(
        ksize=ksize,
        mask=mask_gans_trn,
        mask_sar=mask_sar,
        labels=labels,
        output_folder=patch_trn,
        sar=sar,
        opt=opt,
        stride=226,
        show=True)

    # extract_patches_stride_case_A(
    #     ksize,
    #     mask=mask_gans_trn,
    #     mask_sar=mask_sar,
    #     output_folder=patch_trn,
    #     sar=sar,
    #     opt=opt,
    #     stride=226,
    #     show=True)


def create_dataset_case_C(ksize=256,
                          dataset=None,
                          mask_path=None,
                          sar_path_t0=None,
                          opt_path_t0=None,
                          sar_path_t1=None,
                          opt_path_t1=None,
                          region=None,
                          num_patches=400,
                          show=False):

    patch_trn = '/mnt/Data/Pix2Pix_datasets/' + dataset + '_' + str(ksize) + '/train/'
    root_path = '/mnt/Data/DataBases/RS/'
    labels_path = root_path + 'Quemadas/AP2_Acre/clip_reference_raster_new.tif'
    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)

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
    mask_sar = sar_t0[:, :, 1].copy()
    mask_sar[sar_t0[:, :, 0] < 1] = 1
    mask_sar[sar_t0[:, :, 0] == 1] = 0

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

    sar_t0 = minmaxnormalization(sar_t0, mask_sar)
    sar_t1 = minmaxnormalization(sar_t1, mask_sar)
    opt_t0 = minmaxnormalization(opt_t0, mask_opt)
    opt_t1 = minmaxnormalization(opt_t1, mask_sar)

    extract_patches_stride_case_C(
        ksize=128, # cambiar despues
        mask=mask_gans_trn,
        mask_sar=mask_sar,
        output_folder=patch_trn,
        sar_t0=sar_t0,
        sar_t1=sar_t1,
        opt_t0=opt_t0,
        opt_t1=opt_t1,
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
    mask_gan[(mask != 0) * (mask_gan == 0) ] = 1

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


def create_dataset_4_classifier(ksize=256,
                                dataset=None,
                                mask_path=None,
                                sar_path=None,
                                opt_path=None,
                                region=None,
                                num_patches=400,
                                show=False):

    patch_trn = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+dataset+'/train/classifier/'

    if not os.path.exists(patch_trn):
        os.makedirs(patch_trn)

    mask, sar, opt, cloud_mask = load_images(mask_path=None,
                                             sar_path=sar_path,
                                             opt_path=opt_path)
    labels_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/'
    labels_name = '10_May_2016.tif'
    labels = load_tiff_image(labels_root_path + labels_name)
    labels = resampler(labels)

    cloud_mask[cloud_mask != 0] = 1

    sar = resampler(sar)
    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 0].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0
    mask = resampler(mask)
#    mask_gan = resampler(mask_gan)
    mask_gan = np.load('mask_gan.npy')
    mask_gan[mask == 0] = 0
    mask_gan[(mask != 0) * (mask_gan != 1)] = 2

    mask_gans_trn = mask_gan * mask_sar
    plt.figure()
    plt.imshow(mask_gans_trn)
    plt.show(block=False)
    # plt.figure()
    # plt.imshow(cloud_mask)
    # plt.show(block=True)

    sar = minmaxnormalization(sar, mask_sar)
    opt = minmaxnormalization(opt, mask_gans_trn)

    extract_patches_4_classifier(
        ksize=ksize,
        mask=mask_gans_trn,
        mask_sar=mask_sar,
        labels=labels,
        output_folder=patch_trn,
        sar=sar,
        opt=opt)
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