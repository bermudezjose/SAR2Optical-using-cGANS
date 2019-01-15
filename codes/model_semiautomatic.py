from __future__ import division
import os
import time
import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from sklearn import preprocessing as pre
import scipy.io as io
import matplotlib.pyplot as plt
import keras

from ops import *
from utils import *
from saveweigths import *

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=11, output_c_dim=7, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None, n_features=7,
                 n_classes=8, n_hidden=14):
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
        self.batch_size_classifier = 32
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.beta = 3.0
        self.sample_size = sample_size
        self.output_size = output_size
        self.sar_root_patch = '/mnt/Data/DataBases/RS/SAR/Campo Verde/npy_format/'
        self.opt_root_patch = '/mnt/Data/DataBases/RS/SAR/Campo Verde/LANDSAT/'
        self.sar_name = '14_31Jul_2016.npy'

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

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

        self.data_classifier = tf.placeholder(tf.float32,
                                              [None, self.n_features], name='data_classifier')

        self.labels_classifier = tf.placeholder(tf.int32,
                                                [None, self.n_classes], name='labels_classifier')

        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.C, self.C_logits = self.classifier(self.data_classifier, self.dropout_keep_prob)

        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                         name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]


        self.fake_B = self.generator(self.real_A)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
#        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.c_loss = self.beta * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.C_logits, labels=self.labels_classifier))
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) 
        self.g_lossl1 = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        self.g_loss = self.g_loss0 + self.g_lossl1 + self.c_loss

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.c_vars = [var for var in t_vars if 'c_' in var.name]

        self.saver = tf.train.Saver()


#    def load_random_samples(self):
#        data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
#        sample = [load_data(sample_file) for sample_file in data]
#
#        if (self.is_grayscale):
#            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
#        else:
#            sample_images = np.array(sample).astype(np.float32)
#        return sample_images

#    def sample_model(self, sample_dir, epoch, idx):
#        sample_images = self.load_random_samples()
#        samples, d_loss, g_loss = self.sess.run(
#            [self.fake_B_sample, self.d_loss, self.g_loss],
#            feed_dict={self.real_data: sample_images}
#        )
#        save_images(samples, [self.batch_size, 1],
#                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
#        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        c_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.c_loss, var_list=self.c_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

#        self.g_sum = tf.summary.merge([self.d__sum,
#            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
#        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        data = glob.glob('/mnt/Data/DataBases/Experiments_Semi/05may2016_C01_semi/train/*.npy')
        data_classifier_trn_list = glob.glob('/mnt/Data/DataBases/Experiments_Semi/05may2016_C01_semi/train/classifier/Training/*.npy')
        data_classifier_val_list = glob.glob('/mnt/Data/DataBases/Experiments_Semi/05may2016_C01_semi/train/classifier/Validation/*.npy')
        data_classifier_generator = glob.glob('/mnt/Data/DataBases/Experiments_Semi/05may2016_C01_semi/train/classifier/trn_val/*.npy')
        labels2new_labels = np.load('labels2new_labels.npy').item()

        # opt_root_patch = '/mnt/Data/DataBases/RS/SAR/Campo Verde/LANDSAT/'
        # img = opt_root_patch + '20160505/'
        # labels = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/10_May_2016.tif'
        # mask = 'mask_gan_original.tif'
        # out_trn_data, out_trn_labels, out_val_data, out_val_labels = create_training_samples_Classifier(img, labels, mask)

        # val_samples, val_labels_one_hot = load_data4Validation(self, data_classifier_val_list, labels2new_labels)

        # labels_GANs_Region = np.float32(np.load('trn_labels_classifier.npy'))
        # samples_GANs_Region = np.array(np.load('trn_samples_classifier.npy')).astype(np.float32)
        # idxc = range(len(data_classifier_list))
        # batch_c = 1
        improvement_threshold = 0.995
        isTrain = True
        trn_c_loss = []
        best_eva_err = np.inf
        val_counter = 0
        for epoch in xrange(args.epoch):
            errorD = []
            errorG = []
            errorC = []
            Gl1 = []
            Gc = []
            trn_C_loss_epoch = []
            # print (data[0])
            # print ("num of samples --->", len(data))
            np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                g_loss0_trn = []
                g_lossl1_trn = []
                c_loss_g = []

                batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_images = [load_data(batch_file) for batch_file in batch_files] # Load files from path
                # print ("shape of samples --->", np.shape(batch_images))

                # Update C network
                if (isTrain):
                    _, trn_samples, trn_labels_one_hot, _, _ = load_data4Classifier(self,
                                                                                    data_classifier_trn_list,
                                                                                    labels2new_labels)
                    print("classifier has been updated ...")
                    _, c_loss_C = self.sess.run([c_optim, self.c_loss],
                        feed_dict={self.data_classifier: trn_samples, self.labels_classifier: trn_labels_one_hot, self.dropout_keep_prob: 0.8})
                    # print ("trn c_loss--->", c_loss)
                    trn_C_loss_epoch.append(c_loss_C)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.real_data: batch_images})
                self.writer.add_summary(summary_str, counter)

                sar_patch, trn_samples, trn_labels_one_hot, labels_patch, idx_samples = load_data4Classifier(self,
                                                                                                             data_classifier_generator,
                                                                                                             labels2new_labels)
                g_sample = self.sess.run(self.fake_B_sample,
                                         feed_dict={self.real_A: sar_patch.reshape(1, 256, 256, 2)})
                g_sample = g_sample.reshape(len(labels_patch), self.n_features)
                g_sample = g_sample[idx_samples]

                # Update G network
                _, g_loss0, g_lossl1, c_loss = self.sess.run([g_optim, self.g_loss0, self.g_lossl1, self.c_loss],
                    feed_dict={self.real_data: batch_images, self.data_classifier: g_sample, self.labels_classifier: trn_labels_one_hot, self.dropout_keep_prob: 1.0})
                self.writer.add_summary(summary_str, counter)
                g_loss0_trn.append(g_loss0)
                g_lossl1_trn.append(g_lossl1)
                c_loss_g.append(c_loss)

                g_sample = self.sess.run(self.fake_B_sample,
                                         feed_dict={self.real_A: sar_patch.reshape(1, 256, 256, 2)})
                g_sample = g_sample.reshape(len(labels_patch), self.n_features)
                g_sample = g_sample[idx_samples]
                # print(g_sample.shape, trn_samples.shape)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, g_loss0, g_lossl1, c_loss = self.sess.run([g_optim, self.g_loss0, self.g_lossl1, self.c_loss],
                    feed_dict={self.real_data: batch_images, self.data_classifier: g_sample, self.labels_classifier: trn_labels_one_hot, self.dropout_keep_prob: 1.0})
                self.writer.add_summary(summary_str, counter)
                g_loss0_trn.append(g_loss0)
                g_lossl1_trn.append(g_lossl1)
                c_loss_g.append(c_loss)
                # print(g_sample.shape, trn_samples.shape)
                val_samples, val_labels_one_hot = load_data4Validation(self,
                                                                       data_classifier_val_list,
                                                                       labels2new_labels)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images, self.data_classifier: g_sample, self.labels_classifier: trn_labels_one_hot, self.dropout_keep_prob: 1.0})
                if (isTrain):
                    errC = self.c_loss.eval({self.data_classifier: val_samples, self.labels_classifier: val_labels_one_hot, self.dropout_keep_prob: 1.0})
                    errorC.append(errC)
                    print ("trn c_loss--->", c_loss_C)
                    print ("c_loss Validation: --->", errC)
                errorD.append(errD_fake + errD_real)
                errorG.append(errG)

                if val_counter < 20:
                    if improvement_threshold * best_eva_err > errC:
                        val_counter = 0
                        best_eva_err = errC
                        self.save(args.checkpoint_dir, counter)
                    else:
                        val_counter += 1
                        print(best_eva_err, errC, val_counter)
                elif (isTrain):
                    isTrain = False
                    if self.load(self.checkpoint_dir):
                        print(" [*] Load SUCCESS model with best classifier")
                    else:
                        print(" [!] Load failed...")

                    # print ("The classifier is freezing")
                Gl1.append(np.mean(g_lossl1_trn))
                Gc.append(np.mean(c_loss_g))
                if np.mod(idx, 1) == 0:
                    # print ("trn c_loss -->", c_loss_C)
                    print ("gloss0 -->", np.mean(g_loss0_trn), "glossl1 -->", np.mean(g_lossl1_trn), "closs_g -->", np.mean(c_loss_g))
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (counter, idx, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG))

#                if np.mod(counter, 100) == 1:
#                    self.sample_model(args.sample_dir, epoch, idx)
#                self.save(args.checkpoint_dir, counter)
            print("Mean Errors-->", "D", np.mean(errorD), "G", np.mean(errorG), "Gl1", np.mean(Gl1), "Gc", np.mean(Gc))
            # print("Mean loss classifier training -->", np.mean(trn_C_loss_epoch))
            self.save(args.checkpoint_dir, counter)
            trn_c_loss.append(trn_C_loss_epoch)
            counter += 1


    def classifier(self, features, dropout_keep_prob):
        with tf.variable_scope("classifier") as scope:
            out_layer = mlp(input=features, n_features=self.n_features, n_classes=self.n_classes, n_hidden=self.n_hidden, dropout_keep_prob=dropout_keep_prob)
            return tf.nn.softmax(out_layer), out_layer


    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            # image, _, _ = deconv2d(tf.nn.relu(d1),
                # [self.batch_size, s64, s64, self.input_c_dim], name='g_e0_deconv', with_w=False) 
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv') # 64x2x5x5+64 = 3,264
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')) # (2x64)x64x5x5 + (2x64) = 204,928
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')) # (4x64)x(2x64)x5x5 + (4x64) = 819,456
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')) # (8x64)x(4x64)x5x5 + (8x64) = 3,277,312
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv')) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv')) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv')) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True) # (2*4x64)x(8x64)x5x5 + (2*4x64) = 6,554,112
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True) # (2*2x64)x(4x64)x5x5 + (2*2x64) = 1,638,656
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True) # (2*1x64)x(2x64)x5x5 + (2*1x64) = 409,728
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True) # (1*1x64)x(1x7)x5x5 + (1*1x7) = 11,207
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        print "Saving checkpoint!"
#        self.saver.save(self.sess, checkpoint_dir +'/my-model')
#        self.saver.export_meta_graph(filename=checkpoint_dir +'/my-model.meta')

    def load(self, checkpoint_dir):
#        return False
        print(" [*] Reading checkpoint...")
#
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print(checkpoint_dir)
#2832, 2665,
#        new_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, 2831, 2665,
#                                         self.input_c_dim + self.output_c_dim], name='inputs_new_name')
#        self.saver = tf.train.import_meta_graph(checkpoint_dir +'/my-model.meta', input_map={"real_A_and_B_images:0": new_placeholder})
##        self.saver = tf.train.import_meta_graph(checkpoint_dir +'/my-model.meta')
#        self.saver.restore(self.sess, checkpoint_dir +'/my-model')
#
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
#            self.saver.export_meta_graph(filename='my-model.meta')
#            print 'model convertion success'
#            self.saver = tf.import_graph_def(os.path.join(checkpoint_dir, ckpt_name), input_map={"real_A_and_B_images:0": new_placeholder})
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = sorted(glob.glob('/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/datasets/'+self.dataset_name+'/test/*.npy'))

        # change this directoty

        # sort testing input
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.npy')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]


        # load testing input
        print("Loading testing images ...")
        sample_images = [load_data(sample_file, is_test=True) for sample_file in sample_files]

#        if (self.is_grayscale):
#            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
#        else:
#            sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            print samples.shape
            output_folder = '/home/jose/Templates/'
            np.save(output_folder+str(i), samples.reshape(256, 256, 7))
#            save_images(samples, [self.batch_size, 1],
#                        './{}/test_{:04d}.png'.format(args.test_dir, idx))
    def generate_image(self, args):
        print args
        output_folder = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/'
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(" [*] Load SUCCESS")
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print args.experiment_type
        if args.experiment_type is 'case_A':
            # Load images
            print 'Case 01 ...'
            if '11nov2015' in args.dataset_name:
                print 'generating image for_' + args.dataset_name
                sar_img_name = '02_10Nov_2015.npy'
                opt_img_name = '20151111'
                print sar_img_name, opt_img_name
            elif '13dec2015' in args.dataset_name:
                print 'generating image for_' + args.dataset_name
                sar_img_name = '05_16Dec_2015.npy'
                opt_img_name = '20151213'
                print sar_img_name, opt_img_name
            elif '18mar2016' in args.dataset_name:
                print 'generating image for_' + args.dataset_name
                sar_img_name = '09_21Mar_2016.npy'
                opt_img_name = '20160318'
                print sar_img_name, opt_img_name
            elif '05may2016' in args.dataset_name:
                print 'generating image for_' + args.dataset_name
                sar_img_name = '10_08May_2016.npy'
                opt_img_name = '20160505'
                print sar_img_name, opt_img_name
            elif '08jul2016' in args.dataset_name:
                print 'generating image for_' + args.dataset_name
                sar_img_name = '13_07Jul_2016.npy'
                opt_img_name = '20160708'
                print sar_img_name, opt_img_name
            elif '24jul2016' in args.dataset_name:
                print 'generating image for_' + args.dataset_name
                sar_img_name = '14_31Jul_2016.npy'
                opt_img_name = '20160724'
                print sar_img_name, opt_img_name
            else:
                print "Image pair doesnt exist !!!"
                return 0

            print 'generating image for_' + args.dataset_name
            mask, sar, opt, cloud_mask = load_images(sar_path=self.sar_root_patch + sar_img_name,
                                                     opt_path=self.opt_root_patch + opt_img_name + '/'
                                                     )
            sar = resampler(sar)
            sar[sar > 1.0] = 1.0
            mask_sar = sar[:, :, 0].copy()
            mask_sar[sar[:, :, 0] < 1] = 1
            mask_sar[sar[:, :, 0] == 1] = 0
            mask = resampler(mask)
            mask_gan = np.load('mask_gan.npy')
            mask_gan[mask == 0] = 0
            mask_gan[(mask != 0) * (mask_gan != 1)] = 2
            mask_gans_trn = mask_gan * mask_sar

            sar = minmaxnormalization(sar, mask_sar)
            num_rows, num_cols, num_bands = sar.shape

            img_source = sar.reshape(1, num_rows, num_cols, num_bands)
            fake_opt = np.zeros((num_rows, num_cols, self.output_c_dim),
                                dtype='float32')
            s = 64
            stride = self.image_size-2*s
            for row in range(0, num_rows, stride):
                for col in range(0, num_cols, stride):
                    if (row+self.image_size <= num_rows) and (col+self.image_size <= num_cols):

                        print row + s, row + self.image_size - s
                        sample_image = img_source[:, row:row+self.image_size, col:col+self.image_size]
                        sample = self.sess.run(self.fake_B_sample,
                                               feed_dict={self.real_A: sample_image}
                                               )
                        print sample.shape
                        fake_opt[row+s:row+self.image_size-s, col+s:col+self.image_size-s] = sample[0, s:self.image_size-s, s:self.image_size-s]
                    elif col+self.image_size <= num_cols:
                        sample_image = img_source[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
                        print(sample_image.shape)
                        sample = self.sess.run(self.fake_B_sample,
                                               feed_dict={self.real_A: sample_image}
                                               )
                        print sample.shape
                        fake_opt[row+s:num_rows, col+s:col+self.image_size-s] = sample[0, self.image_size-num_rows+row+s:self.image_size, s:self.image_size-s]
                    elif row+self.image_size <= num_rows:
                        print col
                        sample_image = img_source[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
                        sample = self.sess.run(self.fake_B_sample,
                                               feed_dict={self.real_A: sample_image}
                                               )
                        fake_opt[row+s:row+self.image_size-s, col+s:num_cols] = sample[0, s:self.image_size-s, self.image_size-num_cols+col+s:self.image_size]

            np.save(self.dataset_name + '_fake_opt_classifier', fake_opt)

#         elif args.experiment_type is 'case_C':
#             # Load images
#             print 'Case C ...'
#             print 'generating image for_' + args.dataset_name
#             sar_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/20160909.npy'
#             sar_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20170731/20170731.npy'
#             opt_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/'
#             opt_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20170731/'
# #            opt_path='/mnt/Datos/Datasets/Quemadas/AP2_Acre/Sentinel2/',
# #            sar_img_name = '10_08May_2016.npy'
#             opt_t0 = load_sentinel2(opt_path_t0)
#             opt_t1 = load_sentinel2(opt_path_t1)
#             opt_t0[np.isnan(opt_t0)] = 0
#             opt_t1[np.isnan(opt_t1)] = 0
#             print opt_t0.shape
#             print opt_t1.shape
#             sar_t0 = np.load(sar_path_t0)
#             sar_t1 = np.load(sar_path_t1)
#             mask_gans_trn = 'ap2_train_test_mask.npy'
#             mask_gans_trn = np.load(mask_gans_trn)
#             mask_gans_trn = np.float32(mask_gans_trn)
#             mask_gans_trn[mask_gans_trn == 0] = 1.
#             mask_gans_trn[mask_gans_trn == 255] = 2.
#             print mask_gans_trn.shape

#             sar_t0[sar_t0 > 1.0] = 1.0
#             sar_t1[sar_t1 > 1.0] = 1.0
#             mask_sar = sar_t0[:, :, 1].copy()
#             mask_sar[sar_t0[:, :, 0] < 1] = 1
#             mask_sar[sar_t0[:, :, 0] == 1] = 0

#             mask_gans_trn = mask_gans_trn * mask_sar

#             sar_t0 = minmaxnormalization(sar_t0, mask_sar)
#             sar_t1 = minmaxnormalization(sar_t1, mask_sar)
#             opt_t0 = minmaxnormalization(opt_t0, mask_gans_trn)
#             opt_t1 = minmaxnormalization(opt_t1, mask_sar)

#             num_rows, num_cols, num_bands = opt_t0.shape
#             img_source = np.concatenate((sar_t0, sar_t1, opt_t1), axis=2)
#             img_source = img_source.reshape(1, num_rows, num_cols, self.input_c_dim)
#             fake_opt = np.zeros((num_rows, num_cols, self.output_c_dim),
#                                 dtype='float32')
#             s = 112
#             stride = self.image_size-2*s
#             for row in range(0, num_rows, stride):
#                 for col in range(0, num_cols, stride):
#                     if (row+self.image_size <= num_rows) and (col+self.image_size <= num_cols):

#                         print row + s, row + self.image_size - s
#                         sample_image = img_source[:, row:row + self.image_size, col:col + self.image_size]
#                         sample = self.sess.run(self.fake_B_sample,
#                                                feed_dict={self.real_A: sample_image}
#                                                )
#                         print sample.shape
#                         fake_opt[row+s:row+self.image_size-s, col+s:col+self.image_size-s ]= sample[0, s:self.image_size-s, s:self.image_size-s]
#                     elif col+self.image_size <= num_cols:
#                         sample_image = img_source[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
#                         print(sample_image.shape)
#                         sample = self.sess.run(self.fake_B_sample,
#                                                feed_dict={self.real_A: sample_image}
#                                                )
#                         print sample.shape
#                         fake_opt[row+s:num_rows, col+s:col+self.image_size-s] = sample[0, self.image_size-num_rows+row+s:self.image_size, s:self.image_size-s]
#                     elif row+self.image_size <= num_rows:
#                         print col
#                         sample_image = img_source[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
#                         sample = self.sess.run(self.fake_B_sample,
#                                                feed_dict={self.real_A: sample_image}
#                                                )
#                         fake_opt[row+s:row+self.image_size-s, col+s:num_cols] = sample[0, s:self.image_size-s, self.image_size-num_cols+col+s:self.image_size]

#             np.save(self.dataset_name + '_fake_opt_new', fake_opt)


    def create_dataset(self, args):
        if '11nov2015' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name = '02_10Nov_2015.npy'
            opt_img_name = '20151111/'
            print sar_img_name, opt_img_name
        elif '13dec2015' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name = '05_16Dec_2015.npy'
            opt_img_name = '20151213/'
            print sar_img_name, opt_img_name
        elif '18mar2016' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name = '09_21Mar_2016.npy'
            opt_img_name = '20160318/'
            print sar_img_name, opt_img_name
        elif '05may2016' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name = '10_08May_2016.npy'
            opt_img_name = '20160505/'
            print sar_img_name, opt_img_name
        elif '08jul2016' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name = '13_07Jul_2016.npy'
            opt_img_name = '20160708/'
            print sar_img_name, opt_img_name
        elif '24jul2016' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name = '14_31Jul_2016.npy'
            opt_img_name = '20160724/'
            print sar_img_name, opt_img_name
        elif 'May05_Jul08' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name2 = '10_08May_2016.npy'
            opt_img_name2 = '20160505/'
            sar_img_name1 = '13_07Jul_2016.npy'
            opt_img_name1 = '20160708/'
        elif '13jul2017_C03' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name = '/mnt/Data/DataBases/CampoVerde2017/Sentinel1/20170714.npy'
            opt_img_name = '/mnt/Data/DataBases/CampoVerde2017/Sentinel2/20170713/'
            print sar_img_name, opt_img_name
        elif '08jul2016_multiresolution' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            sar_img_name = '13_07Jul_2016.npy'
            opt_img_name = '20160708/'
            print sar_img_name, opt_img_name
        elif 'quemadas_ap2_case_A' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            create_dataset_case_A(
                ksize=256,
                dataset=self.dataset_name,
                mask_path=None,
                sar_path='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/20160909.npy',
                opt_path='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/'
            )
        elif 'quemadas_ap2_case_C' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            create_dataset_case_C(
                ksize=256,
                dataset=self.dataset_name,
                mask_path=None,
                sar_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/new_20160909.npy',
                sar_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20170731/20170731.npy',
                opt_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/',
                opt_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20170731/'
            )
            
        else:
            print "Image pair doesnt exist !!!"
        print 'creating dataset for_' + args.dataset_name
        create_dataset_4_classifier(
            ksize=256,
            dataset=self.dataset_name,
            mask_path=None,
            sar_path=self.sar_root_patch + sar_img_name,
            opt_path=self.opt_root_patch + opt_img_name
        )
        # return 0
        # create_dataset_case1(ksize=256,
        #                     dataset=self.dataset_name,
        #                     mask_path=None,
        #                     sar_path=self.sar_root_patch + sar_img_name,
        #                     opt_path=self.opt_root_patch + opt_img_name,
        #                     # region=args.image_region,
        #                     num_patches = 1000,
        #                     show=True)
#   20151111 -- 02_10Nov_2015 ok !
#   20151127 -- 03_22Nov_2015 too much clouds !
#   20160318 -- 09_21Mar_2016 ok !
#   20151213 -- 05_16Dec_2015 ummmm, differen protocol ...
#   20160505 -- 10_08May_2016 ok !
#   20160708 -- 13_07Jul_2016 ok !
#   20160724 -- 14_31Jul_2016 ok !
        # create_dataset_case_A(
        #     ksize=256,
        #     dataset=self.dataset_name,
        #     mask_path=None,
        #     sar_path='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/20160909.npy',
        #     opt_path='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/'
        #     )
        # create_dataset_case_C(ksize=256,
        #     dataset=self.dataset_name,
        #     mask_path=None,
        #     sar_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/20160909.npy',
        #     sar_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20170731/20170731.npy',
        #     opt_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/',
        #     opt_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20170731/'
        #     )
#        create_dataset_case3(ksize=256,
#                             dataset=self.dataset_name,
#                             mask_path=None,
#                             sar_path=sar_img_name,
#                             opt_path=opt_img_name,
#                             num_patches = 1000,
#                             show=True)
#        create_dataset_case4(ksize=256,
#                             dataset=self.dataset_name,
#                             mask_path=None,
#                             sar_path=self.sar_root_patch + sar_img_name,
#                             opt_path=self.opt_root_patch + opt_img_name,
#                             num_patches=1000,
#                             show=True)
#        create_dataset_case5(ksize=256,
#                             dataset=self.dataset_name,
#                             mask_path=None,
#                             sar_path=self.sar_root_patch + sar_img_name,
#                             opt_path=self.opt_root_patch + opt_img_name,
#                             num_patches=1000,
#                             show=True)
        # create_dataset_case_d_multiresolution(ksize=256,
        #                                       dataset=self.dataset_name,
        #                                       mask_path=None,
        #                                       landsat_path=self.opt_root_patch + opt_img_name,
        #                                       sent2_path=self.opt_root_patch + opt_img_name,
        #                                       sent1_path=self.sar_root_patch + sar_img_name,
        #                                       region=None)
        return 0