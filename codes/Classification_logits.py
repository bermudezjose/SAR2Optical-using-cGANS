import numpy as np
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import preprocessing as pp
import timeit
import scipy.io as io
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.util.shape import view_as_windows
from PIL import Image
# import h5py


def extract_patches(img, ksize=3):
    padding = np.zeros(shape=(img.shape[0] + 2*(ksize/2),
                              img.shape[1] + 2*(ksize/2),
                              img.shape[2]), dtype=img.dtype)
    padding[ksize/2:padding.shape[0] - ksize/2, ksize/2:padding.shape[1] - ksize/2, :] = img
    kernel = (ksize, ksize, img.shape[2])
    subimgs = view_as_windows(padding, kernel)
    return subimgs

def balance_data(data, labels, samples_per_class):

    classes = np.unique(labels)
    print classes
    num_total_samples = len(classes)*samples_per_class
    out_labels = np.zeros((num_total_samples), dtype='uint8')
    out_data = np.zeros((num_total_samples, data.shape[1]), dtype='float32')

    k = 0
    for clss in classes:
        clss_labels = labels[labels == clss]
        clss_data = data[labels == clss]
        num_samples = len(clss_labels)
        if num_samples > samples_per_class:
            # Choose samples randomly
            index = range(len(clss_labels))
            index = np.random.choice(index, samples_per_class, replace=False)
            out_labels[k*samples_per_class:k*samples_per_class
                       + samples_per_class] = clss_labels[index]
            out_data[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_data[index]

        else:
            # do oversampling
            index = range(len(clss_labels))
            index = np.random.choice(index, samples_per_class, replace=True)
            out_labels[k*samples_per_class:k*samples_per_class
                       + samples_per_class] = clss_labels[index]
            out_data[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_data[index]
        k += 1
    # Permute samples randomly
    idx = np.random.permutation(out_data.shape[0])
    out_data = out_data[idx]
    out_labels = out_labels[idx]

    return out_data, out_labels


if __name__ == '__main__':

    labels_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/'
    labels_name = '10_May_2016.tif'
    # logits_img_path = '05may2016_monotemporal_multiresolution_dense_discriminator_classifier_crops_synthesized_logits.npy'
    logits_img_path = '05may2016_monotemporal_multiresolution_dense_discriminator_classifier_crops_tanh_synthesized_logits.npy'
    logits_img = np.load(logits_img_path)
    predictions = logits_img.reshape(-1, 9)
    predictions = np.argmax(predictions, axis=1)
    # print(logits_img)

    mask_original = np.load('mask_gan_original.npy')
    labels = load_tiff_image(labels_root_path + labels_name)
    mask_gan = np.zeros_like(mask_original)
    mask_gan[(labels != 0) * (mask_original != 0)] = 2
    mask_gan[(labels != 0) * (mask_original == 0)] = 1
    mask_gan = resampler(mask_gan, 'uint8')
    labels = resampler(labels, 'uint8')
    labels2new_labels, new_labels2labels = labels_look_table(labels_root_path + labels_name)


    plt.figure()
    plt.imshow(labels)
    plt.show(block=False)

    plt.figure()
    plt.imshow(mask_gan)
    plt.show(block=False)

    mask_gan = mask_gan.ravel()
    labels = labels.ravel()
    labels_original = labels.copy()
    mask_gan = mask_gan[labels != 5]
    predictions = predictions[labels != 5]
    labels = labels[labels != 5]

    classes = np.unique(labels)
    new_labels = np.zeros_like(labels)
    for clas in classes:
        if clas != 0:
            new_labels[labels == clas] = labels2new_labels[clas]
    print (np.unique(new_labels))
    labels = new_labels.copy()

    # predictions = predictions[mask_gan != 0]
    # labels = labels[mask_gan != 0]
    # mask_gan = mask_gan[mask_gan != 0]

    idtrn = mask_gan == 1
    idtst = mask_gan == 2

    # Labels
    trn_labels = labels[idtrn].copy()
    tst_labels = labels[idtst].copy()

    predictions_trn = predictions[idtrn]
    predictions_tst = predictions[idtst]

    # accuracy
    accuracy = accuracy_score(tst_labels, predictions_tst)
    f1 = 100*f1_score(tst_labels, predictions_tst, average=None)
    rs = 100*recall_score(tst_labels, predictions_tst, average=None)
    ps = 100*precision_score(tst_labels, predictions_tst, average=None)
    print 'f1 score'
    print (np.around(f1, decimals=1))
#    print (np.around(f1.sum(), decimals=1))
    print 'recall'
    print (np.around(rs, decimals=1))
    print ('precision')
    print (np.around(ps, decimals=1))
#    print (np.around(rs.sum(), decimals=1))
    print 'accuracy ->', 100*accuracy
    print 'recall ->', rs.mean()
    print 'precision ->', ps.mean()
    print 'f1 score ->', f1.mean()
    plt.show(block=True)
    
