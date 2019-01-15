#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:31:44 2018

@author: jose
"""
import tensorflow as tf
import numpy as np
from ops import *

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=11, output_c_dim=7, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.sar_root_patch = '/mnt/Data/DataBases/RS/SAR/Campo Verde/npy_format/'
        self.opt_root_patch = '/mnt/Data/DataBases/RS/SAR/Campo Verde/LANDSAT/'
        self.sar_name = '14_31Jul_2016.npy'

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        # batch normalization : deals with poor initialization helps gradient flow

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.input_image = tf.placeholder(tf.float32,
                                          [self.batch_size, self.image_size[0], self.image_size[1],
                                           self.input_c_dim],
                                           name='input_image')
        self.te8 = tf.placeholder(tf.float32,
                                          [self.batch_size, 1, 1, 512],
                                          name='te8')
        self.te7 = tf.placeholder(tf.float32,
                                          [self.batch_size, 2, 2, 512],
                                          name='te7')
        self.te6 = tf.placeholder(tf.float32,
                                          [self.batch_size, 4, 4, 512],
                                          name='te6')
        self.te5 = tf.placeholder(tf.float32,
                                          [self.batch_size, 8, 8, 512],
                                          name='te5')
        self.te4 = tf.placeholder(tf.float32,
                                          [self.batch_size, 16, 16, 512],
                                          name='te4')
        self.te3 = tf.placeholder(tf.float32,
                                          [self.batch_size, 32, 32, 256],
                                          name='te3')
        self.te2 = tf.placeholder(tf.float32,
                                          [self.batch_size, 64, 64, 128],
                                          name='te2')
        self.te1 = tf.placeholder(tf.float32,
                                          [self.batch_size, 128, 128, 64],
                                          name='te1')

        self.output_image = self.generator(self.input_image)

        self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8 = self.downsampler(self.input_image)

        self.upsampling = self.upsampler(self.te1, self.te2, self.te3, self.te4, self.te5, self.te6, self.te7, self.te8)


    def downsampler(self, image, y=None):
        with tf.variable_scope("downsampler") as scope:
#            reuse=tf.AUTO_REUSE
            weigths = np.load("weigths.npy").item()
            We1 = weigths['We1']
            We2 = weigths['We2']
            We3 = weigths['We3']
            We4 = weigths['We4']
            We5 = weigths['We5']
            We6 = weigths['We6']
            We7 = weigths['We7']
            We8 = weigths['We8']

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv', weight_params=We1[0], bias_params=We1[1]) # 64x4x4x2+64 = 2112
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv', weight_params=We2[0], bias_params=We2[1])) # 131072
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv', weight_params=We3[0], bias_params=We3[1])) # 524000
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv', weight_params=We4[0], bias_params=We4[1])) # 2097152
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv', weight_params=We5[0], bias_params=We5[1])) # 4192304
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv', weight_params=We6[0], bias_params=We6[1])) # 4192304
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv', weight_params=We7[0], bias_params=We7[1])) # 4192304
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv', weight_params=We8[0], bias_params=We8[1])) # 4192304

            return e1, e2, e3, e4, e5, e6, e7, e8


    def upsampler(self, e1, e2, e3, e4, e5, e6, e7, e8, y=None):
        with tf.variable_scope("upsampler") as scope:
#            reuse=tf.AUTO_REUSE
            weigths = np.load("weigths.npy").item()
            Wb1 = weigths['Wb1']
            Wb2 = weigths['Wb2']
            Wb3 = weigths['Wb3']
            Wb4 = weigths['Wb4']
            Wb5 = weigths['Wb5']
            Wb6 = weigths['Wb6']
            Wb7 = weigths['Wb7']
            Wb8 = weigths['Wb8']

            r = 256
            r2, r4, r8, r16, r32, r64, r128 = int(r/2), int(r/4), int(r/8), int(r/16), int(r/32), int(r/64), int(r/128)

            c = 256
            c2, c4, c8, c16, c32, c64, c128 = int(c/2), int(c/4), int(c/8), int(c/16), int(c/32), int(c/64), int(c/128)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, r128, c128, self.gf_dim*8], name='g_d1', with_w=True, weight_params=Wb1[0], bias_params=Wb1[1])
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, r64, c64, self.gf_dim*8], name='g_d2', with_w=True, weight_params=Wb2[0], bias_params=Wb2[1])
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, r32, c32, self.gf_dim*8], name='g_d3', with_w=True, weight_params=Wb3[0], bias_params=Wb3[1])
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, r16, c16, self.gf_dim*8], name='g_d4', with_w=True, weight_params=Wb4[0], bias_params=Wb4[1])
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, r8, c8, self.gf_dim*4], name='g_d5', with_w=True, weight_params=Wb5[0], bias_params=Wb5[1])
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, r4, c4, self.gf_dim*2], name='g_d6', with_w=True, weight_params=Wb6[0], bias_params=Wb6[1])
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, r2, c2, self.gf_dim], name='g_d7', with_w=True, weight_params=Wb7[0], bias_params=Wb7[1])
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, r, c, self.output_c_dim], name='g_d8', with_w=True, weight_params=Wb8[0], bias_params=Wb8[1])
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:
#            reuse=tf.AUTO_REUSE
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

            r = self.output_size[0]
            r2, r4, r8, r16, r32, r64, r128 = int(r/2), int(r/4), int(r/8), int(r/16), int(r/32), int(r/64), int(r/128)

            c = self.output_size[1]
            c2, c4, c8, c16, c32, c64, c128 = int(c/2), int(c/4), int(c/8), int(c/16), int(c/32), int(c/64), int(c/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv', weight_params=We1[0], bias_params=We1[1]) # 64x4x4x2+64 = 2112
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv', weight_params=We2[0], bias_params=We2[1])) # 131072
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv', weight_params=We3[0], bias_params=We3[1])) # 524000
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv', weight_params=We4[0], bias_params=We4[1])) # 2097152
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv', weight_params=We5[0], bias_params=We5[1])) # 4192304
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv', weight_params=We6[0], bias_params=We6[1])) # 4192304
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv', weight_params=We7[0], bias_params=We7[1])) # 4192304
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv', weight_params=We8[0], bias_params=We8[1])) # 4192304
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, r128, c128, self.gf_dim*8], name='g_d1', with_w=True, weight_params=Wb1[0], bias_params=Wb1[1])
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, r64, c64, self.gf_dim*8], name='g_d2', with_w=True, weight_params=Wb2[0], bias_params=Wb2[1])
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, r32, c32, self.gf_dim*8], name='g_d3', with_w=True, weight_params=Wb3[0], bias_params=Wb3[1])
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, r16, c16, self.gf_dim*8], name='g_d4', with_w=True, weight_params=Wb4[0], bias_params=Wb4[1])
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, r8, c8, self.gf_dim*4], name='g_d5', with_w=True, weight_params=Wb5[0], bias_params=Wb5[1])
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, r4, c4, self.gf_dim*2], name='g_d6', with_w=True, weight_params=Wb6[0], bias_params=Wb6[1])
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, r2, c2, self.gf_dim], name='g_d7', with_w=True, weight_params=Wb7[0], bias_params=Wb7[1])
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, r, c, self.output_c_dim], name='g_d8', with_w=True, weight_params=Wb8[0], bias_params=Wb8[1])
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def run(self, img):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        output_image = self.sess.run([self.output_image],
                                     feed_dict={ self.input_image: img })
        return output_image

    def down_sampling(self, img):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        e1, e2, e3, e4, e5, e6, e7, e8 = self.sess.run([self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8],
                                                       feed_dict={self.input_image: img })
        return e1, e2, e3, e4, e5, e6, e7, e8


    def up_sampling(self, e1, e2, e3, e4, e5, e6, e7, e8):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        upsampling = self.sess.run([self.upsampling],
                                   feed_dict={ self.te1: e1, self.te2:e2, self.te3:e3, self.te4:e4, self.te5:e5, self.te6:e6, self.te7:e7, self.te8:e8})
        return upsampling

if __name__ == '__main__':
    from sklearn import preprocessing as pre
    sar = np.load('/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/08jul2016_C01_13_07Jul_2016.npy_sar_nor.npy')
    real = np.load("08jul2016_C01_opt.npy")
    num_rows, num_cols, bands = real.shape
    real = real.reshape(num_rows * num_cols, bands)
    scaler = pre.MinMaxScaler((-1, 1)).fit(real)
    real = np.float32(scaler.transform(real))
    real = real.reshape(num_rows, num_cols, bands)

    shape = (256, 256)

    i = 1000
    sar = sar[i:i+shape[0], i:i+shape[1]]
    real = np.float32(real[i:i+shape[0], i:i+shape[1]])
    img_mosaic = np.zeros_like(real)

    shape = sar.shape
#    im = img.copy()
#    img = img[:, :, 0:2]
    sar = sar.reshape(1, shape[0], shape[1], shape[2])
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = pix2pix(sess, image_size=(shape[0], shape[1]), batch_size=1,
                        output_size=(shape[0], shape[1]), input_c_dim=2,
                        output_c_dim=7)
        output_image = model.run(sar)
        ef1, ef2, ef3, ef4, ef5, ef6, ef7, ef8 = model.down_sampling(sar)
        for i in range(shape[0]/256):
            for j in range(shape[0]/256):
                e1 = ef1[:, i*128:(i+1)*128, j*128:(j+1)*128]
                e2 = ef2[:, i*64:(i+1)*64, j*64:(j+1)*64]
                e3 = ef3[:, i*32:(i+1)*32, j*32:(j+1)*32]
                e4 = ef4[:, i*16:(i+1)*16, j*16:(j+1)*16]
                e5 = ef5[:, i*8:(i+1)*8, j*8:(j+1)*8]
                e6 = ef6[:, i*4:(i+1)*4, j*4:(j+1)*4]
                e7 = ef7[:, i*2:(i+1)*2, j*2:(j+1)*2]
                e8 = ef8[:, i*1:(i+1)*1, j*1:(j+1)*1]

                img_mosaic[i*256:(i+1)*256,j*256:(j+1)*256] = model.up_sampling(e1, e2, e3, e4, e5, e6, e7, e8)[0]

        sess.close()


    x = output_image[0].reshape(shape[0] * shape[1], 7)
    x = scaler.inverse_transform(x)
    x = x.reshape(shape[0], shape[1], 7)
#x = im[:, :, 2:8]

    import matplotlib.pyplot as plt
    from skimage import exposure


    im = real[:, :, [3, 2, 1]].copy()
    im = (im - im.min()) / (im.max() - im.min())

#    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
#    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
#    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
    plt.show(block=False)

    im = x[:, :, [3, 2, 1]].copy()
    im = (im - im.min()) / (im.max() - im.min())
#    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
#    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
#    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
    plt.show(block=False)

    im = img_mosaic.reshape(real.shape)
    im = im[:, :, [3, 2, 1]].copy()
    im = (im - im.min()) / (im.max() - im.min())
#    im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.03)
#    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.03)
#    im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.03)

    plt.figure()
    plt.imshow(im)
    plt.show(block=False)
