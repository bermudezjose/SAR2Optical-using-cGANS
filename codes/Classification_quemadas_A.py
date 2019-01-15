import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import preprocessing as pp
from utils import load_sentinel2
from utils import load_sar
from utils import load_tiff_image
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
#from sklearn.metrics import accuracy_score
from skimage import exposure
import scipy
import scipy.io as sio
from utils import inverse_transform


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
    # real_opt_path = root_path + 'Quemadas/AP2_Acre/Sentinel2/20160825/'
    real_opt_path = root_path + 'Quemadas/AP2_Acre/Sentinel2/20170731/'
    sar_path = root_path + 'Quemadas/AP2_Acre/Sentinel1/20160909/new_20160909.npy'
    labels_path = root_path + 'Quemadas/AP2_Acre/clip_reference_raster_new.tif'


    labels = load_tiff_image(labels_path)
    labels[np.isnan(labels)] = 0
    labels[labels != 0] = 1
    border_mask = np.zeros_like(labels)
    # border_mask[112:border_mask.shape[0]-112, 112:border_mask.shape[1]-112] = 1

    opt = load_sentinel2(real_opt_path)
    opt[np.isnan(opt)] = 0
    num_rows, num_cols, num_bands = opt.shape
    print opt.shape

    # fake_img = np.load('quemadas_ap2_case_A_fake_opt_new.npy')
    # fake_img = np.load('quemadas_ap2_case_A_valset_fake_opt.npy')
#    fake_img = np.load('quemadas_ap2_case_A_semi_L03_class_fake_opt_semi.npy')
#    fake_img = np.load('quemadas_ap2_case_A_semi_L03_all_labeled_fake_opt_semi.npy')
    # fake_name = 'quemadas_monotemporal_original_patches_fake_opt.npy'
    # fake_name = 'quemadas_multitemporal_original_patches_fake_opt.npy'
    # fake_img = np.load('quemadas_multitemporal_original_patches_sar_t1_fake_opt.npy')
    # fake_img = np.load('quemadas_multitemporal_original_patches_opt_t1_fake_opt.npy')
    # fake_img = np.load('model_quemadas_monotemporal_dense_discriminator_fake_opt.npy')
    # fake_name = 'model_quemadas_monotemporal_dense_discriminator_unsuper_fake_opt.npy'
    fake_name = 'model_quemadas_monotemporal_dense_discriminator_super_fake_opt.npy'
    # fake_name = 'model_quemadas_multitemporal_dense_discriminator_super_fake_opt.npy'
    # fake_name = 'model_quemadas_multitemporal_dense_discriminator_super_fake_opt_9.npy'
    # fake_name = 'model_quemadas_multitemporal_dense_discriminator_unsuper_fake_opt.npy'
    fake_img = np.load(fake_name)
    print(fake_name)

    sar = load_sar(sar_path)
    # sio.savemat('20160909.mat', {'sar':sar_t0})
#    scipy.misc.imsave('20160909.mat', sar_t0)
    #sar_t1 = np.load(sar_path_t1)
    mask_gans_trn = load_tiff_image('new_train_test_mask.tif')
#    mask_gans_trn = np.load(mask_gans_trn)
    mask_gans_trn = np.float32(mask_gans_trn)
    mask_gans_trn[mask_gans_trn == 0] = 1.
    mask_gans_trn[mask_gans_trn == 255] = 2.
    print mask_gans_trn.shape

    sar[sar > 1.0] = 1.0
    mask_sar = sar[:, :, 1].copy()
    mask_sar[sar[:, :, 0] < 1] = 1
    mask_sar[sar[:, :, 0] == 1] = 0

    # mask_gans_trn = mask_gans_trn * mask_sar * border_mask
    mask_gans_trn = mask_gans_trn * mask_sar

    # scipy.misc.imsave('ap2_train_test_mask.tif', mask_gans_trn)

#     plt.figure()
#     plt.imshow(mask_gans_trn)
#     # plt.show(block=False)
#     # plt.figure()
# #    plt.imshow(labels, alpha=0.5)
#     plt.show(block=False)

    im = opt[:, :, [2, 1, 0]].copy()
    im = (im - im.min()) / (im.max() - im.min())
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

    plt.figure('real')
    plt.imshow(im)
    # plt.imshow(labels, alpha=0.5)
    # plt.show(block=False)

    # opt = opt.reshape(num_rows * num_cols, num_bands)
    # fake_img = fake_img.reshape(num_rows * num_cols, num_bands)
    # scaler = pp.MinMaxScaler((-1, 1)).fit(opt[mask_gans_trn.ravel() != 0])
    # fake_img = scaler.inverse_transform(fake_img)
    # opt = opt.reshape(num_rows, num_cols, num_bands)
    # fake_img = fake_img.reshape(num_rows, num_cols, num_bands)

    im = fake_img[:, :, [2, 1, 0]].copy()
#     im = inverse_transform(im)
    im = (im - im.min()) / (im.max() - im.min())
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

    plt.figure('fake')
    plt.imshow(im)
    # plt.imshow(labels, alpha=0.5)
    # plt.show(block=False)

# %%

#    plt.savefig(fake_name + '.jpg', dpi=1000)

    mask_gans_trn = mask_gans_trn.ravel()
    opt = opt.reshape(num_rows * num_cols, num_bands)
    fake_img = fake_img.reshape(num_rows * num_cols, num_bands)
    sar = sar.reshape(num_rows * num_cols, sar.shape[2])

    idx_trn = mask_gans_trn == 1
    idx_tst = mask_gans_trn == 2

    labels = labels.ravel()
    # trn_data = opt[idx_trn]
    # tst_data = opt[idx_tst]
#    trn_data = opt_t1[idx_trn]
#    tst_data = opt_t1[idx_tst]
    trn_data = fake_img[idx_trn]
    tst_data = fake_img[idx_tst]
#    trn_data = sar_t0[idx_trn]
#    tst_data = sar_t0[idx_tst]
    trn_labels = labels[idx_trn]
    tst_labels = labels[idx_tst]

    # print(len(labels[labels==1]))
    # print(len(labels[labels==0]))

    trn_data, trn_labels = balance_data(trn_data,
                                        trn_labels,
                                        samples_per_class=900000)
    # [{1:1}, {2:5}, {3:1}, {4:1}]
    # dict = [{0:1}, {1:5}, {3:1}, {4:1}]
    clf = RandomForestClassifier(
                                 n_estimators=250,
                                 max_depth=25,
                                 # class_weight=dict, 
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

    print ('recall')
    print (np.around(rs, decimals=1))
    print ('precision')
    print (np.around(ps, decimals=1))
    print ('f1 score')
    print (np.around(f1, decimals=1))
    print ('accuracy ->', 100 * accuracy)
    print ('recall ->', np.around(rs.mean(), decimals=1))
    print ('precision ->', np.around(ps.mean(), decimals=1))
    print ('f1 score ->', np.around(f1.mean(), decimals=1))
    plt.show(block=True)
