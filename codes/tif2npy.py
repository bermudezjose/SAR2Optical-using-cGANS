#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:29:58 2018

@author: jose
"""
import numpy as np
from osgeo import gdal
band_vh = '/mnt/Data/DataBases/CampoVerde2017/May_VH.tif'
band_vv = '/mnt/Data/DataBases/CampoVerde2017/May_VV.tif'

gdal_header = gdal.Open(band_vh)
band_vh = gdal_header.ReadAsArray()

gdal_header = gdal.Open(band_vv)
band_vv = gdal_header.ReadAsArray()

band_vh = 10.0**(band_vh/10.0)
band_vv = 10.0**(band_vv/10.0)
band_vh = band_vh[:, :7995]
img = np.zeros((8492, 7995, 2), dtype='float32')
img[:, :, 0] = band_vh
img[:, :, 1] = band_vv
np.save('20170520', img)


#from PIL import Image
#gdal_header = gdal.Open(img)
#qb = gdal_header.ReadAsArray()
#qb[np.isnan(qb)] = 0
#qb[qb==8] = 1
#qb[qb==80] = 1
#qb[qb!=1] = 0
#im = Image.fromarray(qb, mode='F')
#im.save(img, "TIFF")
#
#8, 72, 136, 200
#16, 80, 144, 208