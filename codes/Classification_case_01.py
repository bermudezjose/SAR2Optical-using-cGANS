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
    # RGB 432
    # Apr_2016.tif.x
    # 10_May_2016.tif
    # 14_Jul_2016.tif


#11nov2015_C01_crops_20151111_cloud_mask.npy
#11nov2015_C01_crops_20151111_fake_opt.npy
#11nov2015_C01_crops_20151111_opt.npy
#11nov2015_C01_crops_opt.npy
#11nov2015_C01_crops_sar.npy
# 02_Nov_2015.tif

#   20151111 -- 02_10Nov_2015 ok !
#   20151127 -- 03_22Nov_2015 too much clouds !
#   20151213 -- 05_16Dec_2015 ummmm, differen protocol ...
#   20160318 -- 09_21Mar_2016 ok !
#   20160505 -- 10_08May_2016 ok !
#   20160708 -- 13_07Jul_2016 ok !
#   20160724 -- 14_31Jul_2016 ok !

#   02_Nov_2015.tif
#   05_Dec_2015.tif
#   09_Mar_2016.tif
#   10_May_2016.tif
#   14_Jul_2016.tif

    sar_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/npy_format/'
    opt_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/LANDSAT/'
    labels_root_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/'
    exp = 3

#    labels_name = '02_Nov_2015.tif'
#    sar_img_name = '02_10Nov_2015.npy'
#    opt_img_name = '20151111'
#    # fake_name = '11nov2015_C01_crops_20151111_fake_opt.npy'
#    fake_name = '11nov2015_C01_fake_opt.npy'

#    labels_name = '05_Dec_2015.tif'
#    sar_img_name = '05_16Dec_2015.npy'
#    opt_img_name = '20151213'
#    fake_name = '13dec2015_C01_fake_opt.npy'

#    labels_name = '09_Mar_2016.tif'
#    sar_img_name = '09_21Mar_2016.npy'
#    opt_img_name = '20160318'
#    fake_name = '18mar2016_C01_fake_opt.npy'

#    labels_name = '09_Mar_2016.tif'
#    sar_img_name = '09_21Mar_2016.npy'
#    opt_img_name = '20160419'
#    fake_name = '18mar2016_C01_fake_opt.npy'

    labels_name = '10_May_2016.tif'
    sar_img_name = '10_08May_2016.npy'
    opt_img_name = '20160505'
#    opt_img_name = '20170524'

#    fake_name = '05may2016_C01_fake_opt.npy'

#    labels_name = '14_Jul_2016.tif'
#    sar_img_name = '13_07Jul_2016.npy'
#    opt_img_name = '20160708'
    # fake_name = 'May052016May202017_multiresolution_fake_opt_new.npy'
#    fake_name = '05may2016_multiresolution_fake_opt_new.npy'
#    fake_name = 'May052016May2017_S1S2L_fake_opt_new.npy'
#    fake_name = '05may2016_C01_semi_fake_opt_classifier.npy'
#    fake_name = '05may2016_C01_fake_opt_classifier.npy'
#    "May052016May2017_S1S2L_fake_opt_new.npy"
#    fake_name = '05may2016_C01_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_wholeimageL1_10_4_9_11_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_baseline_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_oneclass_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_baseline_L1_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_fake_opt.npy'
    # fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_baseline_fake_opt.npy'
    # fake_name = '05may2016_C01_synthesize_fcn_BASELINE_fake_opt.npy' 
    # fake_name = '05may2016_C01_synthesize_fcn_BASELINE_FINETUNING_fake_opt.npy'
    # fake_name = '05may2016_C01_semantic_discriminator_fake_opt.npy'
    # fake_name = '05may2016_C01_semantic_discriminator_fcn_fake_opt.npy'
    # fake_name = '05may2016_C01_semantic_discriminator_fcn_all_samples_fake_opt.npy'
    # fake_name = '05may2016_multi_resolution_temporal_opt_t2_fake_opt.npy'
    # fake_name = '05may2016_multi_resolution_temporal_sar_t2_fake_opt.npy'
    # fake_name = '05may2016_C01_semantic_discriminator_martha_L2_fake_opt.npy'
    # fake_name = '05may2016_monotemporal_multiresolution_crops_synthesized.npy'
    # fake_name = '05may2016_monotemporal_multiresolution_crops_tanh_synthesized.npy'
    # fake_name = '05may2016_multitemporal_multiresolution_crops_tanh_synthesized.npy'
    fake_name = '05may2016_monotemporal_multiresolution_dense_discriminator_classifier_crops_tanh_synthesized.npy'
    # fake_name = '05may2016_multitemporal_multiresolution_crops_tanh_synthesized.npy'
    # fake_name = '05may2016_multi_resolution_temporal_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_L1_0_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_L1_0_C_0_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_L1_0_C_1_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_baseline_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_beta_1_baseline_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_beta_1_baseline2_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_beta_1_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_beta_1_2_fake_opt.npy'
#    fake_name = '05may2016_C01_synthesize_semisupervised_multitemporal_oneclass_2_fake_opt.npy'

#    labels_name = '14_Jul_2016.tif'
#    sar_img_name = '14_31Jul_2016.npy'
#    opt_img_name = '20160724'
#    fake_name = '24jul2016_C01_fake_opt.npy'
#    plt.close('all')
    print('Fake image --->', fake_name)
    opt_fake = np.load(fake_name)
    print(opt_fake.shape)
    mask, sar, opt_real, _ = load_images(sar_path=sar_root_path + sar_img_name,
                                         opt_path=opt_root_path + opt_img_name + '/'
                                         )
    print (opt_real.shape)
    opt_real[np.isnan(opt_real)] = 0.0
    cloud_mask = np.load('cloud_mask.npy')
#    mask_gan = np.load('mask_gan.npy')
    mask_original = np.load('mask_gan_original.npy')
    mask_gan = np.zeros_like(mask_original)
    mask_gan[(mask != 0) * (mask_original != 0)] = 2
    mask_gan[(mask != 0) * (mask_original == 0)] = 1
    mask_gan = resampler(mask_gan, 'uint8')
    # plt.figure()
    # plt.imshow(mask_gan)
    # plt.show(block=False)
#    im = Image.fromarray(mask_gan)
#    im.save('mask_gan_letter.tif')
#    plt.savefig('mask_gan_original.tif')

    sar = np.float32(sar)
    opt_real = np.float32(opt_real)
    mask = resampler(mask, 'uint8')
    sar = resampler(sar, 'float32')
    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 1].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0

#    j1 = 718
#    j2 = 1297
#    i1 = 908
#    i2 = 1425

#    fig, ax = plt.subplots()
##    plt.figure()
#    sar[:, :, 1] = exposure.equalize_adapthist(sar[:, :, 1], clip_limit=0.02)
#    plt.imshow(sar[i1:i2, j1:j2, 1], cmap='gray')
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    fig.tight_layout()
#    plt.show(block=False)
#    plt.savefig('snip_sar_cv.pdf', dpi=100)
#    mask_gan[mask == 0] = 0
#    mask_gan[(mask != 0) * (mask_gan != 1) ] = 2

    labels = load_tiff_image(labels_root_path + labels_name)
    labels = resampler(labels, 'uint8')
    print (np.unique(labels))
    # plt.pause()

    cloud_mask[cloud_mask == 0] = 1
    cloud_mask[cloud_mask != 1] = 0

#    labels = labels.ravel()
#    eps = 10e-8

    mask = mask * cloud_mask
#    mask_gan = mask_gan * cloud_mask
#    mask2 = mask.copy()
    mask[mask != 0] = 1

#    plt.figure()
#    plt.imshow(mask2)
#    plt.show()
#    plt.savefig('mask.jpg', dpi=1000)
#    mask_trn_tst = np.ones_like(mask) # test: 2, train: 1
#    mask_trn_tst[:int(mask.shape[0]/2), :mask.shape[1]] = 2
#    mask_trn_tst = mask_trn_tst*mask
#    mask2[mask_trn_tst == 1] = 0 # seleciona mitad de imagen a excluir


#    jul2016_800
#    opt_01_jul2016_C03_02_bottom.npy
#    sar_01_jul2016_02_top
#    jul2016_C03_01_bottom
#'   08jul2016_C01_top_opt.npy'
#    08jul2016_C01.5_all_20160724_opt.npy

    im = opt_real[:, :, [3, 2, 1]].copy()
    im = (im - im.min()) / (im.max() - im.min())
##    im = exposure.equalize_adapthist(im)
#
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.04)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.04)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.04)
#    snip = im[i1:i2, j1:j2].copy()

#    snip[:, :, 0] = exposure.equalize_adapthist(snip[:, :, 0], clip_limit=0.03)
#    snip[:, :, 1] = exposure.equalize_adapthist(snip[:, :, 1], clip_limit=0.03)
#    snip[:, :, 2] = exposure.equalize_adapthist(snip[:, :, 2], clip_limit=0.03)


    plt.figure()
    plt.imshow(im)
    plt.show(block=False)
#
#    fig, ax = plt.subplots()
##    plt.figure()
#    plt.imshow(snip)
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    fig.tight_layout()
#    plt.show(block=False)
#    plt.savefig('snip_may2017_real_cv.pdf', dpi=300)
#    plt.savefig('snip_may2016_real_cv.pdf', dpi=300)

    num_rows, num_cols, num_bands = opt_real.shape
#    opt_fake = opt_fake[:num_rows, :num_cols, :]
    opt_real = opt_real.reshape(num_rows*num_cols, num_bands)
    num_rows, num_cols, num_bands = opt_fake.shape
    # opt_fake = opt_fake.reshape(num_rows*num_cols, num_bands)
#    scaler = pp.MinMaxScaler((-1, 1)).fit(opt_real)
#    opt_fake = scaler.inverse_transform(opt_fake)
#    opt_real = opt_real.reshape(num_rows, num_cols, num_bands)
#    opt_fake = opt_fake.reshape(num_rows, num_cols, num_bands)
#    import scipy.misc
#    scipy.misc.imsave('multiresolution_fake_cv.tiff', opt_fake[:,:,[3, 2, 1]])


#    num_rows, num_cols, num_bands = opt_fake.shape

    im = opt_fake[:,:,[3, 2, 1]].copy()
    opt_fake = opt_fake.reshape(num_rows*num_cols, num_bands)
##    im = im[1000:1256,1000:1256]
    im = (im - im.min()) / (im.max() - im.min())
#
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.04)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.04)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.04)
#    snip = im[3*i1:3*i2, 3*j1:3*j2].copy()

#    snip[:, :, 0] = exposure.equalize_adapthist(snip[:, :, 0], clip_limit=0.03)
#    snip[:, :, 1] = exposure.equalize_adapthist(snip[:, :, 1], clip_limit=0.03)
#    snip[:, :, 2] = exposure.equalize_adapthist(snip[:, :, 2], clip_limit=0.03)


    plt.figure()
    plt.imshow(im)
    # plt.show(block=False)
#    fig, ax = plt.subplots()
##    plt.figure()
#    plt.imshow(snip)
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    fig.tight_layout()
#    plt.show(block=False)
#    plt.savefig('snip_may2016_cv_fake_multitemporal.pdf', dpi=300)
#    plt.savefig('snip_may2016_cv_fake_monotemporal.pdf', dpi=300)
#    plt.savefig(fake_name + '.jpg', dpi=1000)
#    kernel = 3
#    opt_real = extract_patches(opt_real, kernel)
#    opt_fake = extract_patches(opt_fake, kernel)
#    opt_real = opt_real.reshape(num_rows, num_cols, num_bands*kernel**2)
#    opt_fake = opt_fake.reshape(num_rows, num_cols, num_bands*kernel**2)
#    opt_real = opt_real.reshape(num_rows * num_cols, num_bands*kernel**2)
#    opt_fake = opt_fake.reshape(num_rows * num_cols, num_bands*kernel**2)
#
#    # Normalization
##    scaler = pp.MinMaxScaler((-1, 1)).fit(opt_fake)
##    opt_fake = scaler.transform(opt_fake)
##    scaler = pp.MinMaxScaler((-1, 1)).fit(opt_real)
##    opt_real = scaler.transform(opt_real)
#
##    ndvi = (opt_fake[:, 3] - opt_fake[:, 2])/(opt_fake[:, 3] + opt_fake[:, 2] + eps)
##    ndvi = ndvi.reshape(num_rows*num_cols, 1)
##    opt_fake = np.concatenate((opt_fake, ndvi), axis=1)
#
#    # Loading SAR image
##    '08jul2016_C01_top_sar.npy'
##    08jul2016_C01.5_all_14_31Jul_2016.npy_sar.npy
#    sar = sar.reshape(num_rows*num_cols, 2)
## %%
#    # Normalization
##    mask_trn_tst = mask_trn_tst.ravel()
##    mask2 = mask2.ravel()
#    mask_gan = mask_gan.ravel()
#
##    idscale = mask_trn_tst != 0
##    idtrn = mask_trn_tst == 1
##    idtst = mask_trn_tst == 2
    mask_gan = mask_gan.ravel()
    labels = labels.ravel()
    labels_original = labels.copy()
    mask_gan = mask_gan[labels != 5]
    opt_real = opt_real[labels != 5]
    opt_fake = opt_fake[labels != 5]
    labels = labels[labels != 5]

    # mask_gan = mask_gan[labels != 10]
    # opt_real = opt_real[labels != 10]
    # opt_fake = opt_fake[labels != 10]
    # labels = labels[labels != 10]

    opt_real = opt_real[mask_gan != 0]
    opt_fake = opt_fake[mask_gan != 0]
    labels = labels[mask_gan != 0]
    mask_gan = mask_gan[mask_gan != 0]

#    idscale = mask_gan != 0
    idtrn = mask_gan == 1
    idtst = mask_gan == 2
#
##    idscale = mask2 != 0
##    idtrn = mask2 == 1
##    idtst = mask2 == 2
#
#    scaler = pp.StandardScaler().fit(opt_real[idtrn])
#    opt_real = np.float32(scaler.transform(opt_real))
#    scaler = pp.StandardScaler().fit(opt_fake[idtrn])
#    opt_fake = np.float32(scaler.transform(opt_fake))
##    scaler = pp.StandardScaler().fit(sar[idscale])
##    sar = np.float32(scaler.transform(sar))
    # Labels
    trn_labels = labels[idtrn].copy()
    tst_labels = labels[idtst].copy()

    trn_data_opt_real = opt_real[idtrn]
    trn_data_opt_fake = opt_fake[idtrn]
#    trn_data_sar = sar[idtrn]
    tst_data_opt_real = opt_real[idtst]
    tst_data_opt_fake = opt_fake[idtst]

#    tst_data_sar = sar[idtst]
#

#
#    # Features
#
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
#    elif exp is 4:
#        # Treina: 1ra metade sar, Teste: 2da metade sar
#        trn_data = trn_data_sar
#        tst_data = tst_data_sar
#    elif exp is 5:
#        # SAR01, SAR02, OPT01, OPT02
#        data = opt_01_real_fake
#        scaler = pp.MinMaxScaler().fit(data)
#        data = scaler.transform(data)
#        trn_data = data[id_trn]
#        tst_data = data[id_tst]


#      Normalization
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
