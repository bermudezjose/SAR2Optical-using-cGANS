import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
    # sar_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/20160909.npy'
    # sar_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20170731/20170731.npy'
    # opt_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/'
    # opt_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20170731/'
    root_path = '/mnt/Data/DataBases/RS/'
    real_opt_path_t0 = root_path + 'Quemadas/AP2_Acre/Sentinel2/20160825/'
    sar_path_t0 = root_path + 'Quemadas/AP2_Acre/Sentinel1/20160909/new_20160909.npy'
    real_opt_path_t1 = root_path + 'Quemadas/AP2_Acre/Sentinel2/20170731/'
    #sar_path_t1 = '/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20170731/20170731.npy'
    labels_path = root_path + 'Quemadas/AP2_Acre/clip_reference_raster_new.tif'
    output_folder = '/home/jose/Drive/PUC/Tesis/PropostaTesis/'

    labels = load_tiff_image(labels_path)
    labels[np.isnan(labels)] = 0
    labels[labels != 0] = 1
    border_mask = np.zeros_like(labels)
    border_mask[112:border_mask.shape[0]-112, 112:border_mask.shape[1]-112] = 1


    opt_t0 = load_sentinel2(real_opt_path_t0)
    opt_t1 = load_sentinel2(real_opt_path_t1)
    opt_t0[np.isnan(opt_t0)] = 0
    opt_t1[np.isnan(opt_t1)] = 0
    rows, cols, bands = opt_t0.shape
    print opt_t0.shape

    # fake_img = np.load('quemadas_ap2_case_C_fake_opt_new.npy')
    # fake_img = np.load('quemadas_multitemporal_original_patches_fake_opt.npy')
    fake_img = np.load('quemadas_monotemporal_original_patches_fake_opt.npy')
#    fake_img = np.load('quemadas_ap2_case_C_new_fake_opt_new.npy')
#    fake_img = np.load('quemadas_ap2_case_C_new_128_fake_opt_new.npy')
#    fake_img = np.load('quemadas_ap2_case_A_fake_opt_new.npy')

    sar_t0 = load_sar(sar_path_t0)
    # sio.savemat('20160909.mat', {'sar':sar_t0})
#    scipy.misc.imsave('20160909.mat', sar_t0)
    #sar_t1 = np.load(sar_path_t1)
    mask_gans_trn = load_tiff_image('new_train_test_mask.tif')
#    mask_gans_trn = np.load(mask_gans_trn)
    mask_gans_trn = np.float32(mask_gans_trn)
    mask_gans_trn[mask_gans_trn == 0] = 1.
    mask_gans_trn[mask_gans_trn == 255] = 2.
    print mask_gans_trn.shape

    sar_t0[sar_t0 > 1.0] = 1.0
    mask_sar = sar_t0[:, :, 1].copy()
    mask_sar[sar_t0[:, :, 0] < 1] = 1
    mask_sar[sar_t0[:, :, 0] == 1] = 0
    j1 = 7000
    j2 = 8200
    i1 = 7200
    i2 = 8000
    # j1 = 5600
    # j2 = 6100
    # i1 = 8200
    # i2 = 8700
    # plt sar image
    fig, ax = plt.subplots()
    sar_t0[:, :, 1] = exposure.equalize_adapthist(sar_t0[:, :, 1], clip_limit=0.02)
    plt.imshow(sar_t0[i1:i2, j1:j2, 1], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show(block=False)
    # plt.savefig('snip_sar_rb_tci.pdf', dpi=300)
    # sar_t0 = []

    mask_gans_trn = mask_gans_trn * mask_sar * border_mask
    # scipy.misc.imsave('ap2_train_test_mask.tif', mask_gans_trn)
#    from collections import OrderedDict
#    cmaps = OrderedDict()
    # cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
    #                     'Dark2', 'Set1', 'Set2', 'Set3',
    #                     'tab10', 'tab20', 'tab20b', 'tab20c']

#    mask_gans_plot = mask_gans_trn.copy()
#    mask_gans_plot[mask_gans_plot==0] = 19
#    mask_gans_plot[mask_gans_plot==2] = 10
#
#    fig, ax = plt.subplots()
##    plt.figure()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#    fig.tight_layout()
#    plt.imshow(mask_gans_plot, cmap = "tab20c")
##    ax.legend()
##    plt.legend(['b', 'r', 'd'], loc='upper right', fontsize='xx-large')
#    mynames = ['training region', 'testing region', 'back']
#    mycmap = plt.cm.tab20c # for example
#    cont = 0
#    for entry in [1, 10]:
#        mycolor = mycmap(entry)
#        plt.plot(0, 0, "-", c=mycolor, label=mynames[cont])
#        cont += 1
#
#    plt.show(block=False)
#    plt.legend(loc='upper right', fontsize='x-large')
##    plt.savefig('/home/jose/Drive/PUC/Tesis/PropostaTesis/rio_branco_trn_tst_region.pdf', dpi=300)


    # plt.figure()
    # plt.imshow(labels, alpha=0.5)
#     plt.show(block=False)

    im = opt_t0[:, :, [2, 1, 0]].copy()
    im = (im - im.min()) / (im.max() - im.min())
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.02)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.02)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.02)
    snip = im[i1:i2, j1:j2].copy()

    # plt.figure()
    # plt.imshow(im)
    # plt.show(block=False)

    fig, ax = plt.subplots()
#    plt.figure()
    plt.imshow(snip)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show(block=False)
    plt.savefig(output_folder + 'snip_aug2016_real_rb_tci.pdf', dpi=300)

#     plt.figure()
#     plt.imshow(im)
# #    plt.imshow(labels, alpha=0.5)
#     plt.show(block=False)

    im = opt_t1[:, :, [2, 1, 0]].copy()
    im = (im - im.min()) / (im.max() - im.min())
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.02)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.02)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.02)
    snip = im[i1:i2, j1:j2].copy()

#     plt.figure()
#     plt.imshow(im)
#     plt.show(block=False)

    fig, ax = plt.subplots()
#    plt.figure()
    plt.imshow(snip)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show(block=False)
    plt.savefig(output_folder + 'snip_aug2017_real_rb.pdf', dpi=300)

    # ndvi = (Nir - Red)/(Nir + Red)
#    num_rows, num_cols, num_bands = opt_t0.shape
#    opt_t0 = opt_t0.reshape(num_rows*num_cols, num_bands)
#    ndvi = (opt_t0[2] - opt_t0[3])/(opt_t0[2] + opt_t0[3])
#    ndvi[np.isnan(ndvi)] = 0
#    opt_t0 = np.concatenate((opt_t0, ndvi), axis=1)
#    fake_img = fake_img.reshape(num_rows*num_cols, num_bands)

#    scaler = pp.MinMaxScaler((-1, 1)).fit(opt_t0[mask_gans_trn.ravel() != 0])
#    fake_img = scaler.inverse_transform(fake_img)
#    opt_t0 = opt_t0.reshape(num_rows, num_cols, num_bands)
#    fake_img = fake_img.reshape(num_rows, num_cols, num_bands)
#    num_rows, num_cols, num_bands = opt_fake.shape

    im = fake_img[:, :, [2, 1, 0]].copy()
#    im = im[1000:1256,1000:1256]
    im = (im - im.min()) / (im.max() - im.min())
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.02)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.02)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.02)
    snip = im[i1:i2, j1:j2].copy()

    plt.figure()
    plt.imshow(im)
    plt.show(block=False)

    fig, ax = plt.subplots()
#    plt.figure()
    plt.imshow(snip)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    plt.show(block=False)
    plt.savefig(output_folder + 'snip_monotemporal_real_rb_tci.pdf', dpi=300)
    # plt.savefig(output_folder + 'snip_multitemporal_real_rb_tci.pdf', dpi=300)

#    opt_t0 = opt_t0.reshape(rows * cols, bands)
#    opt_t1 = opt_t1.reshape(rows * cols, bands)
#    fake_img = fake_img.reshape(rows * cols, bands)
##    sar_t0 = sar_t0.reshape(rows * cols, sar_t0.shape[2])
#
#    mask_gans_trn = mask_gans_trn.ravel()
#    idx_trn = mask_gans_trn == 1
#    idx_tst = mask_gans_trn == 2
#
#    labels = labels.ravel()
##    trn_data = opt_t0[idx_trn]
##    tst_data = opt_t0[idx_tst]
##    trn_data = opt_t1[idx_trn]
##    tst_data = opt_t1[idx_tst]
#    trn_data = fake_img[idx_trn]
#    tst_data = fake_img[idx_tst]
##    trn_data = sar_t0[idx_trn]
##    tst_data = sar_t0[idx_tst]
#    trn_labels = labels[idx_trn]
#    tst_labels = labels[idx_tst]
#
#    trn_data, trn_labels = balance_data(trn_data,
#                                        trn_labels,
#                                        samples_per_class=30000)
#
#    clf = RandomForestClassifier(n_jobs=-1)
##    clf = SVC()
#    clf.fit(trn_data, trn_labels)
#
#    # predict
#    predict_batch = 200000
#    predictions = np.zeros((np.shape(tst_data)[0]))
#    for i in range(0, np.shape(tst_data)[0], predict_batch):
#        predictions[i:i+predict_batch] = clf.predict(tst_data[
#            i:i+predict_batch])
#    print('Predinction finished')
##    end_time = timeit.default_timer()
##    training_time = (end_time - start_time)
##    print(' runtime : %.2fm' % ((training_time) / 60.))
#
#    # accuracy
#    accuracy = accuracy_score(tst_labels, predictions)
#    f1 = 100*f1_score(tst_labels, predictions, average=None)
#    rs = 100*recall_score(tst_labels, predictions, average=None)
#    ps = 100*precision_score(tst_labels, predictions, average=None)
#    print 'f1 score'
#    print (np.around(f1, decimals=1))
##    print (np.around(f1.sum(), decimals=1))
#    print 'recall'
#    print (np.around(rs, decimals=1))
##    print (np.around(rs.sum(), decimals=1))
#    print 'precision'
#    print (np.around(ps, decimals=1))
##    print (np.around(ps.sum(), decimals=1))
#    print 'accuracy ->', 100*accuracy
#    print 'recall ->', rs.mean()
#    print 'precision ->', ps.mean()
#    print 'f1 score ->', f1.mean()