#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:38:56 2018

@author: jose
"""

import numpy as np
import matplotlib.pylab as plt

m = np.random.randint(0,255, (10,10,3))
m90 = np.rot90(m, 1, (0, 1))

plt.figure('original')
plt.imshow(m)
plt.show()

plt.figure('rot90')
plt.imshow(m90)
plt.show()