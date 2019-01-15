import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import preprocessing as pp
from utils import load_sentinel2
from utils import load_sar
from utils import load_tiff_image
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
from sklearn.metrics import accuracy_score
from skimage import exposure
import scipy
import scipy.io as sio
import h5py


def load_gclm_features(path):

    h5file = h5py.File(path, 'r')
    fileHeader = h5file['features']
    features = np.float32(fileHeader[:]).T
    h5file.close()
    return features


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
    # sar_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/20160909.npy'
    # sar_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20170731/20170731.npy'
    # opt_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/'
    # opt_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20170731/'
    # root_path = '/mnt/Data/DataBases/RS/'
    # root_path = '/mnt/Data/Datasets/'
    # sar_path = root_path + 'Quemadas/AP2_Acre/Sentinel1/20160909/20160909.npy'
    # root_path = '/mnt/Data/DataBases/RS/'
    # labels_path = root_path + 'Quemadas/AP2_Acre/clip_reference_raster_new.tif'
    # glcm_features_path = root_path + 'Quemadas/AP2_Acre/Sentinel1/20160909/20160909_sar_glcm_8_.mat'

    root_path = '../Quemadas/'
    sar_path = root_path + 'Sentinel1/20160909/new_20160909.npy'
    labels_path = root_path + 'clip_reference_raster_new.tif'
    glcm_features_path = root_path + 'Sentinel1/20160909/20160909_sar_glcm_8_.mat'

    glcm_features = load_gclm_features(glcm_features_path)

    labels = load_tiff_image(labels_path)
    labels[np.isnan(labels)] = 0
    labels[labels != 0] = 1
    num_rows, num_cols = labels.shape

    sar = load_sar(sar_path)
    mask_gans_trn = load_tiff_image('new_train_test_mask.tif')
    mask_gans_trn = np.float32(mask_gans_trn)
    mask_gans_trn[mask_gans_trn == 0] = 1.
    mask_gans_trn[mask_gans_trn == 255] = 2.
    print mask_gans_trn.shape

    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 1].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0

    mask_gans_trn = mask_gans_trn * mask_sar
    # scipy.misc.imsave('ap2_train_test_mask.tif', mask_gans_trn)

    # plt.figure()
    # plt.imshow(mask_gans_trn)
    # plt.show(block=False)
    # plt.figure()
    # plt.imshow(labels, alpha=0.5)
    # plt.show(block=False)

    mask_gans_trn = mask_gans_trn.reshape(num_rows * num_cols, order='F')

    idx_trn = mask_gans_trn == 1
    idx_tst = mask_gans_trn == 2

    labels = labels.reshape(num_rows * num_cols, order='F')

    trn_data = glcm_features[idx_trn]
    tst_data = glcm_features[idx_tst]
    trn_labels = labels[idx_trn]
    tst_labels = labels[idx_tst]

    trn_data, trn_labels = balance_data(trn_data,
                                        trn_labels,
                                        samples_per_class=900000)

    clf = RandomForestClassifier(
                                 n_estimators=250,
                                 max_depth=25,
                                 n_jobs=-1)
    clf.fit(trn_data, trn_labels)

    # predict
    predict_batch = 200000
    predictions = np.zeros((np.shape(tst_data)[0]))
    for i in range(0, np.shape(tst_data)[0], predict_batch):
        predictions[i:i+predict_batch] = clf.predict(tst_data[
            i:i+predict_batch])
    print('Predinction finished')
#    end_time = timeit.default_timer()
#    training_time = (end_time - start_time)
#    print(' runtime : %.2fm' % ((training_time) / 60.))

    # accuracy
    accuracy = accuracy_score(tst_labels, predictions)
    f1 = 100 * f1_score(tst_labels, predictions, average=None)
    rs = 100 * recall_score(tst_labels, predictions, average=None)
    ps = 100 * precision_score(tst_labels, predictions, average=None)
    print 'f1 score'
    print (np.around(f1, decimals=1))
    print (np.around(f1.sum(), decimals=1))
    print 'recall'
    print (np.around(rs, decimals=1))
    print (np.around(rs.sum(), decimals=1))
    print 'accuracy ->', 100 * accuracy
    print 'recall ->', rs.mean()
    print 'precision ->', ps.mean()
    print 'f1 score ->', f1.mean()
