import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import median
import matplotlib.pyplot as plt



img_path = '/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/20160909.npy'
img = np.load(img_path)
img[img > 1] = 1
plt.figure()
plt.imshow(img[:, :, 0])
plt.show()


med = median(img[:, :, 0], disk(5))
plt.figure()
plt.imshow(med)
plt.show()
med = (med/255.0)
img_dif = med-img[:, :, 0]
img_dif[img_dif > -0.9] = 0
img_dif[img_dif != 0] = 1
plt.figure()
plt.imshow(img_dif)
plt.show()

img[img_dif == 1, 0] = med[img_dif == 1]
img[img_dif == 1, 1] = med[img_dif == 1]
np.save('/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/new_20160909.npy', img)
