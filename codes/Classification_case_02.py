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
#import h5py


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
    # RGB 432
    # Apr_2016.tif.x
    # 10_May_2016.tif
    # 14_Jul_2016.tif
#    20160708
#    20160724

    labels_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/14_Jul_2016.tif'
    labels = load_tiff_image(labels_path)
    labels = resampler(labels)
    labels = labels.ravel()
    eps = 10e-8

    mask = np.load('Results/mask.npy')
    mask2 = mask.copy()
    mask[mask != 0] = 1
    plt.close('all')
#    plt.figure()
#    plt.imshow(mask)
#    plt.show()
#    plt.savefig('mask.jpg', dpi=1000)
    mask_trn_tst = np.ones_like(mask) # test: 2, train: 1
    mask_trn_tst[:int(mask.shape[0]/2), :mask.shape[1]] = 2
    mask_trn_tst = mask_trn_tst*mask
#    mask2[mask_trn_tst == 1] = 0 # seleciona mitad de imagen a excluir
#    plt.figure()
#    plt.imshow(mask2)
#    plt.show()

#    jul2016_800
#    opt_01_jul2016_C03_02_bottom.npy
#    sar_01_jul2016_02_top
#    jul2016_C03_01_bottom

    opt_real = np.load('08jul2016_C02_opt.npy').astype('float32')

    im = opt_real[:, :, [3, 2, 1]].copy()
    im = im/im.max()
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
    plt.show()
#    plt.savefig('08jul2016.jpg', dpi=1000)

    # Computing NDVI
    num_rows, num_cols, num_bands = opt_real.shape
    opt_real = opt_real.reshape(num_rows*num_cols, num_bands)
#    opt_01_real_part = opt_real.copy()
#    ndvi = (opt_real[:, 3] - opt_real[:, 2])/(opt_real[:, 3] + opt_real[:, 2]+eps)
#    ndvi = ndvi.reshape(num_rows*num_cols, 1)
#    opt_real = np.concatenate((opt_real, ndvi), axis=1)

    opt_fake = np.load('08jul2016_C02_fake_opt.npy')
    im = opt_fake[:,:,[3, 2, 1]].copy()
    im = (im - im.min()) / (im.max() - im.min())
    im[:,:,0] = exposure.equalize_adapthist(im[:,:,0], clip_limit=0.03)
    im[:,:,1] = exposure.equalize_adapthist(im[:,:,1], clip_limit=0.03)
    im[:,:,1] = exposure.equalize_adapthist(im[:,:,2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
    plt.show()
#    plt.savefig('24jul2016_fake_bottom.jpg', dpi=1000)

    opt_fake = opt_fake.reshape(num_rows*num_cols, num_bands)
#    ndvi = (opt_fake[:, 3] - opt_fake[:, 2])/(opt_fake[:, 3] + opt_fake[:, 2] + eps)
#    ndvi = ndvi.reshape(num_rows*num_cols, 1)
#    opt_fake = np.concatenate((opt_fake, ndvi), axis=1)

    # Loading SAR image
    sar = np.load('08jul2016_C02_sar.npy')
    sar = sar.reshape(num_rows*num_cols, 2)
# %%
    # Normalization
    mask_trn_tst = mask_trn_tst.ravel()
    mask2 = mask2.ravel()
    idscale = mask_trn_tst != 0
    scaler = pp.StandardScaler().fit(opt_real[idscale])
    opt_real = np.float32(scaler.transform(opt_real))
    scaler = pp.StandardScaler().fit(opt_fake[idscale])
    opt_fake = np.float32(scaler.transform(opt_fake))
    scaler = pp.StandardScaler().fit(sar[idscale])
    sar = np.float32(scaler.transform(sar))

#    idtrn = mask_trn_tst == 1
#    idtst = mask_trn_tst == 2
    idtrn = mask2 == 1
    idtst = mask2 == 2
    trn_data_opt_real = opt_real[idtrn]
    trn_data_opt_fake = opt_fake[idtrn]
    trn_data_sar = sar[idtrn]
    tst_data_opt_real = opt_real[idtst]
    tst_data_opt_fake = opt_fake[idtst]
    tst_data_sar = sar[idtst]

    # Labels
    trn_labels = labels[idtrn].copy()
    tst_labels = labels[idtst].copy()

    # Features
    exp = 2
    if exp is 1:
        # Treina: 1ra metade otico real, Teste: 2da metade otico real (baseline 1)
        trn_data = trn_data_opt_real
        tst_data = tst_data_opt_real
    elif exp is 2:
        # Treina: 1ra metade otico real, Teste: 2da metade otico fake
        trn_data = trn_data_opt_real
        tst_data = tst_data_opt_fake
    elif exp is 3:
        # Treina: 1ra metade otico fake, Teste: 2da metade otico fake
        trn_data = trn_data_opt_fake
        tst_data = tst_data_opt_fake
    elif exp is 4:
        # Treina: 1ra metade sar, Teste: 2da metade sar
        trn_data = trn_data_sar
        tst_data = tst_data_sar
    elif exp is 5:
        # SAR01, SAR02, OPT01, OPT02
        data = opt_01_real_fake
        scaler = pp.MinMaxScaler().fit(data)
        data = scaler.transform(data)
        trn_data = data[id_trn]
        tst_data = data[id_tst]


    #  Normalization
#    scaler = pp.StandardScaler().fit(trn_data)
#    trn_data = scaler.transform(trn_data)
#    tst_data = scaler.transform(tst_data)

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
    print 'f1 score'
    print (np.around(f1, decimals=1))
    print (np.around(f1.sum(), decimals=1))
    print 'recall'
    print (np.around(rs, decimals=1))
    print (np.around(rs.sum(), decimals=1))
    print 'accuracy ->', 100*accuracy
    print 'recall ->', rs.mean()
    print 'precision ->', ps.mean()
    print 'f1 score ->', f1.mean()
