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
#import scipy
import scipy.io as sio
from utils import inverse_transform
from skimage import filters



if __name__ == '__main__':
    root_path = '/mnt/Data/DataBases/RS/'
    real_opt_path = root_path + 'Quemadas/AP2_Acre/Sentinel2/20160825/'
    sar_path = root_path + 'Quemadas/AP2_Acre/Sentinel1/20160909/new_20160909.npy'
    labels_path = root_path + 'Quemadas/AP2_Acre/clip_reference_raster_new.tif'

    labels = load_tiff_image(labels_path)
    labels[np.isnan(labels)] = 0
    labels[labels != 0] = 1

    opt = load_sentinel2(real_opt_path)
    opt[np.isnan(opt)] = 0
    num_rows, num_cols, num_bands = opt.shape
    print opt.shape

    im = opt[:, :, [2, 1, 0]].copy()
    im = (im - im.min()) / (im.max() - im.min())
    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
    im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
#    plt.imshow(labels, alpha=0.5)
    plt.show(block=False)

    plt.figure()
    plt.imshow(labels)
    plt.show(block=False)

    ndvi = (opt[:, :, 2]-opt[:, :, 3])/(opt[:, :, 2]+opt[:, :, 3])
    ndvi[np.isnan(ndvi)] = 0
    plt.figure()
    plt.imshow(ndvi)
    plt.show(block=False)

    val = filters.threshold_otsu(ndvi)
    mask = ndvi < val
    plt.figure()
    plt.imshow(mask)
    plt.show(block=False)

    val = filters.threshold_otsu(opt)
    mask = opt < val
    plt.figure()
    plt.imshow(mask[:,:, 3])
    plt.show(block=False)
