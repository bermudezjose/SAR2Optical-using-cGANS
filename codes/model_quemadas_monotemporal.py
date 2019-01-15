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

from ops import *
from utils import *
# from saveweigths import *

class pix2pix(object):
    def __init__(self, sess, image_size=128,
                 batch_size=1, sample_size=1, output_size=128,
                 load_size=143, fine_size=128, 
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=2, output_c_dim=4, dataset_name='facades',
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
        self.load_size = load_size
        self.fine_size = fine_size
        self.sar_root_patch = '/mnt/Data/DataBases/RS/SAR/Campo Verde/npy_format/'
        self.opt_root_patch = '/mnt/Data/DataBases/RS/SAR/Campo Verde/LANDSAT/'
        self.sar_name = '14_31Jul_2016.npy'

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        self.r_bn = batch_norm(name='r_bn')
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
        self.bands_sar = 2
        self.sar = tf.placeholder(tf.float32,
                                  [self.batch_size, self.image_size, self.image_size, self.bands_sar],
                                   name='sar_images')
        self.opt = tf.placeholder(tf.float32,
                                 [self.batch_size, self.image_size, self.image_size, self.output_c_dim],
                                 name='opt_images')

        self.real_A = self.sar
        self.real_B = self.opt

        self.fake_B = self.generator(self.real_A)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
#        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

#        self.g_sum = tf.summary.merge([self.d__sum,
#            self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
#        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 50
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(counter, args.epoch):
            data_original = glob.glob('/mnt/Data2/Quemadas/Training/*.npy')
            # data_original = glob.glob('/mnt/Data/Pix2Pix_datasets/Quemadas/Training/*.npy')
            # data_original = glob.glob('../Quemadas/Training/*.npy')
            # data_fliped = glob.glob('../Quemadas/Training_flip/*.npy')
            # data = data_original + data_fliped
            data = data_original
            np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = load_data_Dic(samples_list=data,
                                             idxc=idx,
                                             load_size=self.load_size,
                                             fine_size=self.fine_size,
                                             random_transformation=True,
                                             multitemporal=False)  
                sar_t0 = batch_images[0].reshape(self.batch_size, self.image_size, self.image_size, -1)
                opt_t0 = batch_images[1].reshape(self.batch_size, self.image_size, self.image_size, -1)
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.sar: sar_t0, self.opt: opt_t0})
                self.writer.add_summary(summary_str, epoch)

                # Update G network
                _ = self.sess.run([g_optim],
                                     feed_dict={ self.sar: sar_t0, self.opt: opt_t0 })
#                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _ = self.sess.run([g_optim],
                                  feed_dict={self.sar: sar_t0, self.opt: opt_t0})
#                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({ self.sar: sar_t0, self.opt: opt_t0 })
                errD_real = self.d_loss_real.eval({ self.sar: sar_t0, self.opt: opt_t0 })
                errG = self.g_loss.eval({ self.sar: sar_t0, self.opt: opt_t0 })

                if np.mod(idx + 1, 99) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG))

            self.save(args.checkpoint_dir, epoch)


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
            s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64)

            # image is (128 x 128 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (64 x 64 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (32 x 32 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (16 x 16 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (8 x 8 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (4 x 4 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (2 x 2 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e7),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e6], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e5], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e4], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e3], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e2], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e1], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s, s, self.output_c_dim], name='g_d7', with_w=True)

            return tf.nn.tanh(self.d7)


    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64)

            # image is (128 x 128 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (64 x 64 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (32 x 32 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (16 x 16 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (8 x 8 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (4 x 4 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (2 x 2 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e7),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e6], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e5], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e4], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e3], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e2], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e1], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s, s, self.output_c_dim], name='g_d7', with_w=True)

            return tf.nn.tanh(self.d7)


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

        output_folder = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/'

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        print(" [*] Load SUCCESS")
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loading images ...
        sar_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/new_20160909.npy'
        sar_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20170731/20170731.npy'
        opt_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/'
        opt_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20170731/'
        sar_t0 = np.load(sar_path_t0)
        sar_t1 = np.load(sar_path_t1)
        opt_t0 = load_sentinel2(opt_path_t0)
        opt_t1 = load_sentinel2(opt_path_t1)
        sar_t0[sar_t0 > 1.0] = 1.0
        sar_t1[sar_t1 > 1.0] = 1.0
        opt_t0[np.isnan(opt_t0)] = 0
        opt_t1[np.isnan(opt_t1)] = 0
        print 'Case A ...'
        print 'generating image for_' + args.dataset_name
        
        # loading scalers ...
        path_dataset = '/mnt/Data/Pix2Pix_datasets/Quemadas/'
        scaler_sar_t0 = joblib.load(path_dataset + "sar_t0_Scaler")
        scaler_opt_t0 = joblib.load(path_dataset + "opt_t0_Scaler")
        scaler_sar_t1 = joblib.load(path_dataset + "sar_t1_Scaler")
        scaler_opt_t1 = joblib.load(path_dataset + "opt_t1_Scaler")         
        num_rows, num_cols, num_bands = sar_t0.shape
        print('sar_t0.shape --->', sar_t0.shape)
        sar_t0 = sar_t0.reshape(num_rows * num_cols, -1)
        sar_t1 = sar_t1.reshape(num_rows * num_cols, -1)
        opt_t0 = opt_t0.reshape(num_rows * num_cols, -1)
        opt_t1 = opt_t1.reshape(num_rows * num_cols, -1)
        sar_t0 = np.float32(scaler_sar_t0.transform(sar_t0))
        sar_t1 = np.float32(scaler_sar_t1.transform(sar_t1))
        opt_t0 = np.float32(scaler_opt_t0.transform(opt_t0))
        opt_t1 = np.float32(scaler_opt_t1.transform(opt_t1))
        SAR_t0 = sar_t0.reshape(1, num_rows, num_cols, -1)
        SAR_t1 = sar_t1.reshape(1, num_rows, num_cols, -1)
        OPT_t0 = opt_t0.reshape(1, num_rows, num_cols, -1)
        OPT_t1 = opt_t1.reshape(1, num_rows, num_cols, -1)
        
        fake_opt = np.zeros((num_rows, num_cols, self.output_c_dim),
                            dtype='float32')
        
        stride = 112
        pad = (self.image_size - stride) // 2
        for row in range(0, num_rows, stride):
            for col in range(0, num_cols, stride):
                if (row + self.image_size <= num_rows) and (col+self.image_size <= num_cols):

                    print row + pad, row + self.image_size - pad
                    sars_t0 = SAR_t0[:, row:row+self.image_size, col:col+self.image_size]
                    opts_t0 = OPT_t0[:, row:row+self.image_size, col:col+self.image_size]
                    sample = self.sess.run(self.fake_B_sample,
                                           feed_dict={self.sar: sars_t0, self.opt: opts_t0}
                                           )
                    print sample.shape
                    fake_opt[row+pad:row+self.image_size-pad, col+pad:col+self.image_size-pad] = sample[0, pad:self.image_size-pad, pad:self.image_size-pad]
                elif col+self.image_size <= num_cols:
                    sars_t0 = SAR_t0[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
                    opts_t0 = OPT_t0[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
                    sample = self.sess.run(self.fake_B_sample,
                                           feed_dict={self.sar: sars_t0, self.opt: opts_t0})                                           
                    print sample.shape
                    fake_opt[row+pad:num_rows, col+pad:col+self.image_size-pad] = sample[0, self.image_size-num_rows+row+pad:self.image_size, pad:self.image_size-pad]
                elif row+self.image_size <= num_rows:
                    print col
                    sars_t0 = SAR_t0[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
                    opts_t0 = OPT_t0[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
                    sample = self.sess.run(self.fake_B_sample,
                                           feed_dict={self.sar: sars_t0, self.opt: opts_t0})  
                    fake_opt[row+pad:row+self.image_size-pad, col+pad:num_cols] = sample[0, pad:self.image_size-pad, self.image_size-num_cols+col+pad:self.image_size]

        ##########################################################################################################################################################
        ############################# UNCOMENT FOR MULTITEMPORAL ##################################
        # for row in range(0, num_rows_sar, stride):
        #     for col in range(0, num_cols, stride):
        #         if (row + self.image_size <= num_rows) and (col+self.image_size <= num_cols):

        #             print row + s, row + self.image_size - s
        #             sars_t0 = SAR_t0[:, row:row+self.image_size, col:col+self.image_size]
        #             opts_t0 = OPT_t0[:, row:row+self.image_size, col:col+self.image_size]
        #             sars_t1 = SAR_t0[:, row:row+self.image_size, col:col+self.image_size]
        #             opts_t1 = OPT_t0[:, row:row+self.image_size, col:col+self.image_size]
        #             sample = self.sess.run(self.fake_B_sample,
        #                                    feed_dict={self.sar_t0: sars_t0, self.opt_t0: opts_t0, self.sar_t1: sars_t1, self.opt_t1: opts_t1})
        #             print sample.shape
        #             fake_opt[row+s:row+self.image_size-s, col+s:col+self.image_size-s] = sample[0, s:self.image_size-s, s:self.image_size-s]
        #         elif col+self.image_size <= num_cols:
        #             sars_t0 = SAR_t0[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
        #             opts_t0 = OPT_t0[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
        #             sars_t1 = SAR_t1[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
        #             opts_t1 = OPT_t1[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
        #             sample = self.sess.run(self.fake_B_sample,
        #                                    feed_dict={self.sar_t0: sars_t0, self.opt_t0: opts_t0, self.sar_t1: sars_t1, self.opt_t1: opts_t1})
        #             print sample.shape
        #             fake_opt[row+s:num_rows, col+s:col+self.image_size-s] = sample[0, self.image_size-num_rows+row+s:self.image_size, s:self.image_size-s]
        #         elif row+self.image_size <= num_rows:
        #             print col
        #             sars_t0 = SAR_t0[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
        #             opts_t0 = OPT_t0[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
        #             sars_t1 = SAR_t1[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
        #             opts_t1 = OPT_t1[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
        #             sample = self.sess.run(self.fake_B_sample,
        #                                    feed_dict={self.sar_t0: sars_t0, self.opt_t0: opts_t0, self.sar_t1: sars_t1, self.opt_t1: opts_t1})
        #             fake_opt[row+s:row+self.image_size-s, col+s:num_cols] = sample[0, s:self.image_size-s, self.image_size-num_cols+col+s:self.image_size]

        np.save(self.dataset_name + '_fake_opt', fake_opt)


    def create_dataset(self, args):
        create_dataset_Quemandas_Multitemporal(ksize=128,
                                               dataset=self.dataset_name,
                                               mask_path=None,
                                               sar_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/new_20160909.npy',
                                               sar_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20170731/20170731.npy',
                                               opt_path_t0='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20160825/',
                                               opt_path_t1='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel2/20170731/'
                                               )
        # data_augmentation()
        return 0