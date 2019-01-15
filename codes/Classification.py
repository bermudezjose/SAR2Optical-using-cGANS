import numpy as np
from utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import preprocessing as pp
import timeit
import scipy.io as io
import h5py


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
    # Apr_2016.tif.x
    # 10_May_2016.tif
    # 14_Jul_2016.tif
#    20160708
#    20160724
    glcm = False
    labels_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/14_Jul_2016.tif'
    labels = load_tiff_image(labels_path)
    labels = resampler(labels)
    eps = 10e-7

    real_opt = np.load('real_opt_jul2016.npy')
    num_rows, num_cols, num_bands = real_opt.shape
    real_opt = real_opt.reshape(num_rows*num_cols, num_bands)
    ndvi = (real_opt[:, 3] - real_opt[:, 2])/(real_opt[:, 3] + real_opt[:, 2])
    ndvi = ndvi.reshape(num_rows*num_cols, 1)
    real_opt = np.concatenate((real_opt, ndvi), axis=1)

    fake_opt = np.load('fake_opt_jul2016.npy')
    fake_opt = fake_opt.reshape(num_rows*num_cols, num_bands)
    ndvi = (fake_opt[:, 3] - fake_opt[:, 2])/(fake_opt[:, 3] + fake_opt[:, 2] + eps)
    ndvi = ndvi.reshape(num_rows*num_cols, 1)
    fake_opt = np.concatenate((fake_opt, ndvi), axis=1)
    # fake_opt = np.load('fake_opt.npy')
    # real_opt = np.load('real_opt.npy')

#    sar = np.load('sar_jul2016.npy')
    mask = np.load('mask.npy')

    mask_50 = np.ones_like(mask)
    mask_50[:int(mask.shape[0]/2), :mask.shape[1]] = 0

    mask_50 = mask_50[mask != 0]
    fake_opt = fake_opt[mask.ravel() != 0]
    real_opt = real_opt[mask.ravel() != 0]
#    sar = sar[mask != 0]

    if glcm:
        labels = labels.reshape(labels.shape[0] * labels.shape[1], order='F')
        mask = mask.reshape(mask.shape[0] * mask.shape[1], order='F')
        h5file = h5py.File('sar_glcm_8_.mat')
        fileHeader = h5file['features']
        sar_glcm = np.float32(fileHeader[:]).T
        h5file.close()

    labels = np.uint8(labels[mask != 0])
    mask = mask[mask != 0]

    #  #####  Classification using RF
    id_trn = mask == 1
    id_tst = mask == 2

    mask_50_trn = mask_50[id_trn]
    mask_50_tst = mask_50[id_tst]

    # Labels
    trn_labels = labels[id_trn].copy()
    trn_labels = trn_labels[mask_50_trn == 1]
    tst_labels = labels[id_tst].copy()
    tst_labels = tst_labels[mask_50_tst == 0]

    # Features
    trn_data = real_opt[id_trn]
    trn_data = trn_data[mask_50_trn == 1]
#    tst_data = real_opt[id_tst]
#    tst_data = tst_data[mask_50_tst == 0]

    tst_data = fake_opt[id_tst]
    tst_data = tst_data[mask_50_tst == 0]

#    trn_data = sar[id_trn]
#    trn_data = trn_data[mask_50_trn == 1]
#    tst_data = sar[id_tst]
#    tst_data = tst_data[mask_50_tst == 0]

    #trn_data = sar_glcm[id_trn].copy()
    #tst_data = sar_glcm[id_tst]


    #  Normalization
    scaler = pp.StandardScaler().fit(trn_data)
    trn_data = scaler.transform(trn_data)
    tst_data = scaler.transform(tst_data)



    trn_data, trn_labels = balance_data(trn_data,
                                        trn_labels,
                                        samples_per_class=30000)

    print 'trnset_data tensor size --> ', trn_data.shape
    print 'testdata tensor size --> ', tst_data.shape

    # SKLEARN  Classifier
    start_time = timeit.default_timer()
    clf = RandomForestClassifier(n_estimators=250,
                                 max_depth=25,
                                 n_jobs=-1)
    print('Start training...')
    clf = clf.fit(trn_data, trn_labels)
    print('Training finished')

    # predict
    predict_batch = 200000
    predictions = np.zeros((np.shape(tst_data)[0]))
    for i in range(0, np.shape(tst_data)[0], predict_batch):
        predictions[i:i+predict_batch] = clf.predict(tst_data[
            i:i+predict_batch])
    print('Predinction finished')
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print(' runtime : %.2fm' % ((training_time) / 60.))

    # accuracy
    accuracy = accuracy_score(tst_labels, predictions)
    f1 = 100*f1_score(tst_labels, predictions, average=None)
    rs = 100*recall_score(tst_labels, predictions, average=None)
    ps = 100*precision_score(tst_labels, predictions, average=None)
    print (np.around(f1, decimals=1))
    print (np.around(f1.sum(), decimals=1))
    print 'recall'
    print (np.around(rs, decimals=1))
    print (np.around(rs.sum(), decimals=1))
    print ('accuracy ->', 100*accuracy)
    print ('recall ->', rs.mean())
    print ('precision ->', ps.mean())
    print ('f1 score ->', f1.mean())
