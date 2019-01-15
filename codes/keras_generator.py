#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:02:12 2018

@author: jose
"""
import numpy as np
#import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, Conv2DTranspose, Activation, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
from sklearn import preprocessing as pre

#saver = tf.train.import_meta_graph('/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/checkpoint/08jul2016_C01_1_256/pix2pix.model-2.meta')
##graph = tf.get_default_graph()
#init_op = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init_op)
#sess.graph.get_operations()
#e1 = sess.graph.get_tensor_by_name('generator/g_e1_conv/w:0')
#eb1 = sess.graph.get_tensor_by_name('generator/g_e1_conv/biases:0')
#w = e1.eval(session=sess)
#b = eb1.eval(session=sess)
#We1 = [w, b]
#e2 = sess.graph.get_tensor_by_name('generator/g_e2_conv/w:0')
#eb2 = sess.graph.get_tensor_by_name('generator/g_e2_conv/biases:0')
#w = e2.eval(session=sess)
#b = eb2.eval(session=sess)
#We2 = [w, b]
##print e2.eval(session=sess).shape
#e3 = sess.graph.get_tensor_by_name('generator/g_e3_conv/w:0')
#eb3 = sess.graph.get_tensor_by_name('generator/g_e3_conv/biases:0')
#w = e3.eval(session=sess)
#b = eb3.eval(session=sess)
#We3 = [w, b]
##print e3.eval(session=sess).shape
#e4 = sess.graph.get_tensor_by_name('generator/g_e4_conv/w:0')
#eb4 = sess.graph.get_tensor_by_name('generator/g_e4_conv/biases:0')
#w = e4.eval(session=sess)
#b = eb4.eval(session=sess)
#We4 = [w, b]
##print e4.eval(session=sess).shape
#e5 = sess.graph.get_tensor_by_name('generator/g_e5_conv/w:0')
#eb5 = sess.graph.get_tensor_by_name('generator/g_e5_conv/biases:0')
#w = e5.eval(session=sess)
#b = eb5.eval(session=sess)
#We5 = [w, b]
##print e5.eval(session=sess).shape
#e6 = sess.graph.get_tensor_by_name('generator/g_e6_conv/w:0')
#eb6 = sess.graph.get_tensor_by_name('generator/g_e6_conv/biases:0')
#w = e6.eval(session=sess)
#b = eb6.eval(session=sess)
#We6 = [w, b]
##print e6.eval(session=sess).shape
#e7 = sess.graph.get_tensor_by_name('generator/g_e7_conv/w:0')
#eb7 = sess.graph.get_tensor_by_name('generator/g_e7_conv/biases:0')
#w = e7.eval(session=sess)
#b = eb7.eval(session=sess)
#We7 = [w, b]
##print e7.eval(session=sess).shape
#e8 = sess.graph.get_tensor_by_name('generator/g_e8_conv/w:0')
#eb8 = sess.graph.get_tensor_by_name('generator/g_e8_conv/biases:0')
#w = e8.eval(session=sess)
#b = eb8.eval(session=sess)
#We8 = [w, b]
##print e8.eval(session=sess).shape
## graph.get_operations()
#
#d1 = sess.graph.get_tensor_by_name('generator/g_d1/w:0')
#db1 = sess.graph.get_tensor_by_name('generator/g_d1/biases:0')
#w = d1.eval(session=sess)
#b = db1.eval(session=sess)
#Wb1 = [w, b]
##print d1.eval(session=sess).shape
##print db1.eval(session=sess).shape
#d2 = sess.graph.get_tensor_by_name('generator/g_d2/w:0')
#db2 = sess.graph.get_tensor_by_name('generator/g_d2/biases:0')
#w = d2.eval(session=sess)
#b = db2.eval(session=sess)
#Wb2 = [w, b]
##print d2.eval(session=sess).shape
#d3 = sess.graph.get_tensor_by_name('generator/g_d3/w:0')
#db3 = sess.graph.get_tensor_by_name('generator/g_d3/biases:0')
#w = d3.eval(session=sess)
#b = db3.eval(session=sess)
#Wb3 = [w, b]
##print d3.eval(session=sess).shape
#d4 = sess.graph.get_tensor_by_name('generator/g_d4/w:0')
#db4 = sess.graph.get_tensor_by_name('generator/g_d4/biases:0')
#w = d4.eval(session=sess)
#b = db4.eval(session=sess)
#Wb4 = [w, b]
##print d4.eval(session=sess).shape
#d5 = sess.graph.get_tensor_by_name('generator/g_d5/w:0')
#db5 = sess.graph.get_tensor_by_name('generator/g_d5/biases:0')
#w = d5.eval(session=sess)
#b = db5.eval(session=sess)
#Wb5 = [w, b]
##print d5.eval(session=sess).shape
#d6 = sess.graph.get_tensor_by_name('generator/g_d6/w:0')
#db6 = sess.graph.get_tensor_by_name('generator/g_d6/biases:0')
#w = d6.eval(session=sess)
#b = db6.eval(session=sess)
#Wb6 = [w, b]
##print d6.eval(session=sess).shape
#d7 = sess.graph.get_tensor_by_name('generator/g_d7/w:0')
#db7 = sess.graph.get_tensor_by_name('generator/g_d7/biases:0')
#w = d7.eval(session=sess)
#b = db7.eval(session=sess)
#Wb7 = [w, b]
##print d7.eval(session=sess).shape
#d8 = sess.graph.get_tensor_by_name('generator/g_d8/w:0')
#db8 = sess.graph.get_tensor_by_name('generator/g_d8/biases:0')
#w = d8.eval(session=sess)
#b = db8.eval(session=sess)
#Wb8 = [w, b]
##print d8.eval(session=sess).shape
#sess.close()


output_size = [256, 256, 7]
input_size = [256, 256, 2]
s = output_size[1]
s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
gf_dim = 64
k = 5

input_data = Input(shape = input_size)
e1 = Conv2D(gf_dim, (k, k), strides=(2, 2), padding='same', name='g_e1_conv')(input_data)
e1 = LeakyReLU(alpha=0.2)(e1)

e2 = Conv2D(gf_dim*2, (k, k), strides=(2, 2), padding='same', name='g_e2_conv')(e1)
e2 = BatchNormalization()(e2)
e2 = LeakyReLU(alpha=0.2)(e2)

e3 = Conv2D(gf_dim*4, (k, k), strides=(2, 2), padding='same', name='g_e3_conv')(e2)
e3 = BatchNormalization()(e3)
e3 = LeakyReLU(alpha=0.2)(e3)

e4 = Conv2D(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_e4_conv')(e3)
e4 = BatchNormalization()(e4)
e4 = LeakyReLU(alpha=0.2)(e4)

e5 = Conv2D(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_e5_conv')(e4)
e5 = BatchNormalization()(e5)
e5 = LeakyReLU(alpha=0.2)(e5)

e6 = Conv2D(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_e6_conv')(e5)
e6 = BatchNormalization()(e6)
e6 = LeakyReLU(alpha=0.2)(e6)

e7 = Conv2D(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_e7_conv')(e6)
e7 = BatchNormalization()(e7)
e7 = LeakyReLU(alpha=0.2)(e7)

e8 = Conv2D(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_e8_conv')(e7)
e8 = BatchNormalization()(e8)
e8 = Activation("relu")(e8)

d1 = Conv2DTranspose(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_d1')(e8)
d1 = BatchNormalization()(d1)
d1 = Concatenate()([d1, e7])
d1 = Activation("relu")(d1)

d2 = Conv2DTranspose(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_d2')(d1)
d2 = BatchNormalization()(d2)
d2 = Concatenate()([d2, e6])
d2 = Activation("relu")(d2)

d3 = Conv2DTranspose(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_d3')(d2)
d3 = BatchNormalization()(d3)
d3 = Concatenate()([d3, e5])
d3 = Activation("relu")(d3)

d4 = Conv2DTranspose(gf_dim*8, (k, k), strides=(2, 2), padding='same', name='g_d4')(d3)
d4 = BatchNormalization()(d4)
d4 = Concatenate()([d4, e4])
d4 = Activation("relu")(d4)

d5 = Conv2DTranspose(gf_dim*4, (k, k), strides=(2, 2), padding='same', name='g_d5')(d4)
d5 = BatchNormalization()(d5)
d5 = Concatenate()([d5, e3])
d5 = Activation("relu")(d5)

d6 = Conv2DTranspose(gf_dim*2, (k, k), strides=(2, 2), padding='same', name='g_d6')(d5)
d6 = BatchNormalization()(d6)
d6 = Concatenate()([d6, e2])
d6 = Activation("relu")(d6)

d7 = Conv2DTranspose(gf_dim, (k, k), strides=(2, 2), padding='same', name='g_d7')(d6)
d7 = BatchNormalization()(d7)
d7 = Concatenate()([d7, e1])
d7 = Activation("relu")(d7)

d8 = Conv2DTranspose(output_size[2], (k, k), strides=(2, 2), padding='same', name='g_d8')(d7)
#d8 = BatchNormalization()(d8)

output = Activation("tanh")(d8)

model = Model(inputs=input_data, outputs=output)

weigths = np.load("weigths.npy").item()
We1 = weigths['We1']
We2 = weigths['We2']
We3 = weigths['We3']
We4 = weigths['We4']
We5 = weigths['We5']
We6 = weigths['We6']
We7 = weigths['We7']
We8 = weigths['We8']

Wb1 = weigths['Wb1']
Wb2 = weigths['Wb2']
Wb3 = weigths['Wb3']
Wb4 = weigths['Wb4']
Wb5 = weigths['Wb5']
Wb6 = weigths['Wb6']
Wb7 = weigths['Wb7']
Wb8 = weigths['Wb8']


model.layers[1].set_weights(We1)
model.layers[3].set_weights(We2)
model.layers[6].set_weights(We3)
model.layers[9].set_weights(We4)
model.layers[12].set_weights(We5)
model.layers[15].set_weights(We6)
model.layers[18].set_weights(We7)
model.layers[21].set_weights(We8)

model.layers[24].set_weights(Wb1)
model.layers[28].set_weights(Wb2)
model.layers[32].set_weights(Wb3)
model.layers[36].set_weights(Wb4)
model.layers[40].set_weights(Wb5)
model.layers[44].set_weights(Wb6)
model.layers[48].set_weights(Wb7)
model.layers[52].set_weights(Wb8)

#adam = Adam(lr=10e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(loss='binary_crossentropy',
#              optimizer=adam,
#              metrics=['accuracy'])

img = np.load('/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/08jul2016_C01/train/2.npy')
im = img.copy()
img = img[:, :, 0:2]
#num_rows, num_cols, bands = img.shape
#img = img.reshape(num_rows * num_cols, bands)
#scaler = pre.MinMaxScaler((-1, 1)).fit(img)
#img = np.float32(scaler.transform(img))
#img = img.reshape(num_rows, num_cols, bands)
img = img.reshape(1, 256, 256, 2)

x = model.predict(img)
x = x.reshape(256, 256, 7)
#x = im[:, :, 2:8]

import matplotlib.pyplot as plt
from skimage import exposure

im = x[:, :, [3, 2, 1]].copy()
im = (im - im.min()) / (im.max() - im.min())
im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

plt.figure()
plt.imshow(im)
plt.show(block=True)
