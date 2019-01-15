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
#    plt.close('all')
#    plt.figure()
#    plt.imshow(mask)
#    plt.show()
#    plt.savefig('mask.jpg', dpi=1000)
    mask_top = np.ones_like(mask)
    mask_top[:int(mask.shape[0]/2), :mask.shape[1]] = 0
    mask_bottom = np.zeros_like(mask)
    mask_bottom[:int(mask.shape[0]/2), :mask.shape[1]] = 1
    mask = mask.ravel()
    mask_top = mask_top.ravel()  # Real:1, Fake:0
    mask_bottom = mask_bottom.ravel()  # Real:0, Fake:1

#    jul2016_800
#    opt_01_jul2016_C03_02_bottom.npy
#    sar_01_jul2016_02_top
#    jul2016_C03_01_bottom

    dataset = 'jul2016_800'
    region = 'top'
    if region is 'top':
        mask_50 = mask_top
        mask_50_z = mask_bottom
    elif region is 'bottom':
        mask_50 = mask_bottom
        mask_50_z = mask_top

#    opt_01 = np.load('opt_01_' + dataset + '.npy').astype('float32')
#    opt_02 = np.load('opt_02_' + dataset + '.npy').astype('float32')  # source
#    sar_01 = np.load('sar_01_' + dataset +'.npy').astype('float32')
#    sar_02 = np.load('sar_02_' + dataset + '.npy').astype('float32')
#
#    im = opt_01[:,:,[3, 2, 1]].copy()
#    im = im/im.max()
#    im[:,:,0] = exposure.equalize_adapthist(im[:,:,0], clip_limit=0.03)
#    im[:,:,1] = exposure.equalize_adapthist(im[:,:,1], clip_limit=0.03)
#    im[:,:,1] = exposure.equalize_adapthist(im[:,:,2], clip_limit=0.03)
#    plt.close('all')
#    plt.figure()
#    plt.imshow(im)
#    plt.show()
##    plt.savefig('08jul2016.jpg', dpi=1000)
#
#    im = opt_02[:,:,[3, 2, 1]]
#    im = im/im.max()
#    im[:,:,0] = exposure.equalize_adapthist(im[:,:,0], clip_limit=0.03)
#    im[:,:,1] = exposure.equalize_adapthist(im[:,:,1], clip_limit=0.03)
#    im[:,:,1] = exposure.equalize_adapthist(im[:,:,2], clip_limit=0.03)
#
#    plt.figure()
#    plt.imshow(im/im.max())
#    plt.show()
##    plt.savefig('24jul2016.jpg', dpi=1000)
#
#    # Computing NDVI
#    num_rows, num_cols, num_bands = opt_01.shape
#    opt_01 = opt_01.reshape(num_rows*num_cols, num_bands)
#    opt_01_real_part = opt_01.copy()
#    ndvi = (opt_01[:, 3] - opt_01[:, 2])/(opt_01[:, 3] + opt_01[:, 2]+eps)
#    ndvi = ndvi.reshape(num_rows*num_cols, 1)
#    opt_01 = np.concatenate((opt_01, ndvi), axis=1)
#    opt_02 = opt_02.reshape(num_rows*num_cols, num_bands)
#    ndvi = (opt_02[:, 3] - opt_02[:, 2])/(opt_02[:, 3] + opt_02[:, 2]+eps)
#    ndvi = ndvi.reshape(num_rows*num_cols, 1)
#    opt_02 = np.concatenate((opt_02, ndvi), axis=1)

#    fake_opt = np.load('fake_opt_' + dataset + '.npy')
    fake_opt = np.load('Results/24jul2016_C01_top_fake_opt.npy')
    im = fake_opt[:,:,[3, 2, 1]].copy()
    im = (im - im.min()) / (im.max() - im.min())
    im[:,:,0] = exposure.equalize_adapthist(im[:,:,0], clip_limit=0.03)
    im[:,:,1] = exposure.equalize_adapthist(im[:,:,1], clip_limit=0.03)
    im[:,:,1] = exposure.equalize_adapthist(im[:,:,2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
    plt.show()
#    plt.savefig('24jul2016_fake_bottom.jpg', dpi=1000)
#    fake_opt = fake_opt.reshape(num_rows*num_cols, num_bands)
#    ndvi = (fake_opt[:, 3] - fake_opt[:, 2])/(fake_opt[:, 3] + fake_opt[:, 2] + eps)
#    ndvi = ndvi.reshape(num_rows*num_cols, 1)
#    fake_opt = np.concatenate((fake_opt, ndvi), axis=1)

# %%
    scaler = pp.MinMaxScaler((-1, 1)).fit(opt_01_real_part[mask_50 == 1])
    opt_01_real_part = np.float32(scaler.transform(opt_01_real_part))
    opt_01_real_fake = opt_01_real_part*mask_50.reshape(len(mask_50), 1) + fake_opt*mask_50_z.reshape(len(mask_50_z), 1) # Image composition
    ndvi = (opt_01_real_fake[:, 3] - opt_01_real_fake[:, 2])/(opt_01_real_fake[:, 3] + opt_01_real_fake[:, 2] + eps)
    ndvi = ndvi.reshape(num_rows*num_cols, 1)
    opt_01_real_fake = np.concatenate((opt_01_real_fake, ndvi), axis=1)

    # Reshape SAR
    _, _, num_bands = sar_01.shape
    sar_01 = sar_01.reshape(num_rows*num_cols, num_bands)
    sar_02 = sar_02.reshape(num_rows*num_cols, num_bands)

    # Deleting background
#    mask_50 = mask_50[mask != 0]
    fake_opt = fake_opt[mask != 0]
    opt_01 = opt_01[mask != 0]
    opt_02 = opt_02[mask != 0]
    sar_01 = sar_01[mask != 0]
    sar_02 = sar_02[mask != 0]
    opt_01_real_fake = opt_01_real_fake[mask != 0]
    labels = np.uint8(labels[mask != 0])
    mask = mask[mask != 0]

    #  #####  Classification using RF
    id_trn = mask == 1
    id_tst = mask == 2

#    mask_50_trn = mask_50[id_trn]
#    mask_50_tst = mask_50[id_tst]

    # Labels
    trn_labels = labels[id_trn].copy()
    tst_labels = labels[id_tst].copy()

    # Features
    exp = 5
    if exp is 1:
        # SAR01, SAR02, OPT02
        data = np.concatenate((sar_01, sar_02, opt_02), axis=1)
#        data = opt_02
        scaler = pp.MinMaxScaler().fit(data)
        data = scaler.transform(data)
        trn_data = data[id_trn]
        tst_data = data[id_tst]
    elif exp is 2:
        # SAR01, SAR02, OPT01, OPT02
        data = np.concatenate((sar_01, sar_02, opt_01, opt_02), axis=1)
#        data = np.concatenate((sar_01, opt_02), axis=1)
        scaler = pp.MinMaxScaler().fit(data)
        data = scaler.transform(data)
        trn_data = data[id_trn]
        tst_data = data[id_tst]
    elif exp is 3:
        # SAR01, SAR02, OPT01, OPT02
        data = np.concatenate((sar_01, sar_02, opt_01_real_fake, opt_02), axis=1)
        scaler = pp.MinMaxScaler().fit(data)
        data = scaler.transform(data)
        trn_data = data[id_trn]
        tst_data = data[id_tst]
    elif exp is 4:
        # SAR01, SAR02, OPT01, OPT02
        data = opt_01
        scaler = pp.MinMaxScaler().fit(data)
        data = scaler.transform(data)
        trn_data = data[id_trn]
        tst_data = data[id_tst]
    elif exp is 5:
        # SAR01, SAR02, OPT01, OPT02
        data = opt_01_real_fake
        scaler = pp.MinMaxScaler().fit(data)
        data = scaler.transform(data)
        trn_data = data[id_trn]
        tst_data = data[id_tst]

#    tst_data = real_opt[id_tst]
#    tst_data = tst_data[mask_50_tst == 0]

#    tst_data = fake_opt[id_tst]
#    tst_data = tst_data[mask_50_tst == 0]

#    trn_data = sar[id_trn]
#    trn_data = trn_data[mask_50_trn == 1]
#    tst_data = sar[id_tst]
#    tst_data = tst_data[mask_50_tst == 0]

    #trn_data = sar_glcm[id_trn].copy()
    #tst_data = sar_glcm[id_tst]


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
#                                 max_depth=25,
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
    print 'accuracy ->', 100*accuracy
    print 'recall ->', rs.mean()
    print 'precision ->', ps.mean()
    print 'f1 score ->', f1.mean()
