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
    plt.close('all')

    sar_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/npy_format/'
    opt_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/LANDSAT/'
    labels_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/'
    exp = 3

    labels_name = '14_Jul_2016.tif'
    sar_img_name = '13_07Jul_2016.npy'
    opt_img_name = '20160708'
    fake_name = '13jul2017_C03_fake_opt_new.npy'
    opt_fake = np.load(fake_name)
    mask, sar, opt_real, _ = load_images(sar_path=sar_root_path + sar_img_name,
                                         opt_path=opt_root_path + opt_img_name + '/'
                                         )
    cloud_mask = np.load('cloud_mask.npy')
#    mask_gan = np.load('mask_gan.npy')
    mask_original = np.load('mask_gan_original.npy')
    mask_gan = np.zeros_like(mask_original)
    mask_gan[(mask != 0) * (mask_original != 0)] = 2
    mask_gan[(mask != 0) * (mask_original == 0)] = 1
    plt.figure()
    plt.imshow(mask_gan)
    plt.show()
#    plt.savefig('mask_gan.jpg', dpi=1000)

    sar = np.float32(sar)
    opt_real = np.float32(opt_real)

#    mask_gan[mask == 0] = 0
#    mask_gan[(mask != 0) * (mask_gan != 1) ] = 2

    labels = load_tiff_image(labels_root_path + labels_name)

    cloud_mask[cloud_mask == 0] = 1
    cloud_mask[cloud_mask != 1] = 0

    labels = labels.ravel()
    eps = 10e-8

#    mask = mask * cloud_mask
#    mask_gan = mask_gan * cloud_mask
    mask2 = mask.copy()
    mask[mask != 0] = 1

    im = opt_real[:, :, [3, 2, 1]].copy()
    im = (im - im.min()) / (im.max() - im.min())
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
    plt.show(block=False)
#    plt.savefig(opt_img_name + '.jpg', dpi=1000)

    num_rows, num_cols, num_bands = opt_real.shape
    opt_real = opt_real.reshape(num_rows*num_cols, num_bands)
#    opt_fake = opt_fake.reshape(num_rows*num_cols, num_bands)
    scaler = pp.MinMaxScaler((-1, 1)).fit(opt_real)
#    opt_fake = scaler.inverse_transform(opt_fake)
#    opt_real = opt_real.reshape(num_rows, num_cols, num_bands)
#    opt_fake = opt_fake.reshape(num_rows, num_cols, num_bands)


#    num_rows, num_cols, num_bands = opt_fake.shape

    im = opt_fake[:,:,[3, 2, 1]].copy()
#    im = im[1000:1256,1000:1256]
    im = (im - im.min()) / (im.max() - im.min())
    im[:,:,0] = exposure.equalize_adapthist(im[:,:,0], clip_limit=0.03)
    im[:,:,1] = exposure.equalize_adapthist(im[:,:,1], clip_limit=0.03)
    im[:,:,1] = exposure.equalize_adapthist(im[:,:,2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
    plt.show(block=False)
#    plt.savefig(fake_name + '.jpg', dpi=1000)

#    kernel = 3
#    opt_real = extract_patches(opt_real, kernel)
#    opt_fake = extract_patches(opt_fake, kernel)
#    opt_real = opt_real.reshape(num_rows, num_cols, num_bands*kernel**2)
#    opt_fake = opt_fake.reshape(num_rows, num_cols, num_bands*kernel**2)
#    opt_real = opt_real.reshape(num_rows * num_cols, num_bands*kernel**2)
#    opt_fake = opt_fake.reshape(num_rows * num_cols, num_bands*kernel**2)
    num_rows_f, num_cols_f, num_bands_f = opt_fake.shape
    opt_fake = opt_fake.reshape(num_rows_f * num_cols_f, num_bands_f)

#    ndvi = (opt_fake[:, 3] - opt_fake[:, 2])/(opt_fake[:, 3] + opt_fake[:, 2] + eps)
#    ndvi = ndvi.reshape(num_rows*num_cols, 1)
#    opt_fake = np.concatenate((opt_fake, ndvi), axis=1)

    # Loading SAR image
#    '08jul2016_C01_top_sar.npy'
#    08jul2016_C01.5_all_14_31Jul_2016.npy_sar.npy
#    sar = sar.reshape(num_rows*num_cols, 2)
# %%
    # Normalization
    mask2 = mask2.ravel()
    mask_gan = mask_gan.ravel()

#    idscale = mask_trn_tst != 0
#    idtrn = mask_trn_tst == 1
#    idtst = mask_trn_tst == 2
    mask_gan = mask_gan.ravel()
    idscale = mask_gan != 0
    idtrn = mask_gan == 1
    idtst = mask_gan == 2

#    idscale = mask2 != 0
#    idtrn = mask2 == 1
#    idtst = mask2 == 2

#    scaler = pp.StandardScaler().fit(opt_real[idscale])
#    opt_real = np.float32(scaler.transform(opt_real))
#    scaler = pp.StandardScaler().fit(opt_fake[idscale])
#    opt_fake = np.float32(scaler.transform(opt_fake))
#    scaler = pp.StandardScaler().fit(sar[idscale])
#    sar = np.float32(scaler.transform(sar))

#    trn_data_opt_real = opt_real[idtrn]
    trn_data_opt_fake = opt_fake[idtrn]
#    trn_data_sar = sar[idtrn]
#    tst_data_opt_real = opt_real[idtst]
    tst_data_opt_fake = opt_fake[idtst]
#    tst_data_sar = sar[idtst]

    # Labels
    trn_labels = labels[idtrn].copy()
    tst_labels = labels[idtst].copy()

    # Features

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
