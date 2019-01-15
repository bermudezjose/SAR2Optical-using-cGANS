from __future__ import division
import os
import time
import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from sklearn import preprocessing as pre
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import scipy.io as io
import matplotlib.pyplot as plt
import keras
import matplotlib.pyplot as plt
from ops import *
from utils import *
from saveweigths import *
from skimage import exposure

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=11, output_c_dim=7, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None, n_features=7,
                 n_classes=8, isTrain=True):
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
        self.dropout_rate = 0.3
        self.isTrain = isTrain
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


        self.fcn_bn_e2 = batch_norm(name='fcn_bn_e2')
        self.fcn_bn_e3 = batch_norm(name='fcn_bn_e3')
        self.fcn_bn_e4 = batch_norm(name='fcn_bn_e4')

        self.fcn_bn_d1 = batch_norm(name='fcn_bn_d1')
        self.fcn_bn_d2 = batch_norm(name='fcn_bn_d2')
        self.fcn_bn_d3 = batch_norm(name='fcn_bn_d3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):


        self.beta = tf.placeholder(tf.float32)
        self.dropout = tf.placeholder(tf.bool)
        self.labels = tf.placeholder(tf.int32,
                                     [self.batch_size, self.image_size, self.image_size],
                                     name='labels')

        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                         name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A)

        self.FCN_logits_real = self.classifier_fcn(self.real_B, reuse=False)
        self.FCN_logits_fake = self.classifier_fcn(self.fake_B, reuse=True)

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
#        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.noise = tf.random_uniform(shape=tf.shape(self.D),
                                       minval=0,
                                       maxval=0.2,
                                       dtype=tf.float32
                                       )

        # Calculate cross entropy
        self.fcn_loss_real, _, _ = cal_loss(self, logits=self.FCN_logits_real, labels=self.labels)
        self.fcn_loss_fake, _, _ = cal_loss(self, logits=self.FCN_logits_fake, labels=self.labels)
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)-self.noise))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)+self.noise))
        self.g_loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)-self.noise)) 
        self.g_lossl1 = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        self.g_loss = self.g_loss0 + self.g_lossl1
        self.g_loss_sup = self.g_loss + self.beta * self.fcn_loss_fake

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.fcn_vars = [var for var in t_vars if 'fcn_' in var.name]
        self.saver = tf.train.Saver()


    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss_sup, var_list=self.g_vars)
        fcn_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                            .minimize(self.fcn_loss_real, var_list=self.fcn_vars)                          

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

        sample_dir_root = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/sample/'
        sample_dir = os.path.join(sample_dir_root, self.dataset_name)

        labels_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/10_May_2016.tif'
        labels2new_labels, new_labels2labels = labels_look_table(labels_path)

        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        # Cambiar para los datos de Campo Verde
        datasets_root = '/mnt/Data/Pix2Pix_datasets/Semi_Exp/'
        dataset_name = '05may2016_C01_synthesize_fcn_BASELINE/'
        data_trn_list = glob.glob(datasets_root + dataset_name + 'Training/*.npy')
        data_val_list = glob.glob(datasets_root + dataset_name + 'Training/*.npy')
        data_test_list= glob.glob(datasets_root + dataset_name + 'Testing/*.npy')

        data_Dic = np.load(data_test_list[6]).item()
        tst_labels = np.array(data_Dic['labels'])
        tst_img_A = np.array(data_Dic['img_A']).astype('float32').reshape(1, 256, 256, self.input_c_dim)
        tst_img_A = np.concatenate((tst_img_A, tst_img_A), axis=0)
        print (np.shape(tst_img_A))
        tst_img_B = np.array(data_Dic['img_B']).astype('float32')
        fig = self.plot_patch(tst_img_B, n_fig="Testing Patch")
        fig.savefig(sample_dir + '/sample_original_tst.png', dpi=300)

        data_Dic = np.load(data_trn_list[6]).item()
        trn_labels = np.array(data_Dic['labels'])
        trn_img_A = np.array(data_Dic['img_A']).astype('float32').reshape(1, 256, 256, self.input_c_dim)
        trn_img_A = np.concatenate((trn_img_A, trn_img_A), axis=0)
        trn_img_B = np.array(data_Dic['img_B']).astype('float32')
        fig = self.plot_patch(trn_img_B, n_fig="Training Patch")
        fig.savefig(sample_dir + '/sample_original_trn.png', dpi=300)

        # plt.figure("Testing Labels")
        # plt.imshow(tst_labels)
        # plt.show(block=False)
        # plt.figure("Training Labels")
        # plt.imshow(trn_labels)
        # plt.show(block=False)
        # plt.pause(0.5)        
        Val_loss = []
        Trn_loss = []
        fig10 = plt.figure("classifier_fcn")
        # for epoch in xrange(args.epoch):
        for epoch in xrange(0):
            loss = []            
            np.random.shuffle(data_trn_list)
            batch_idxs = min(len(data_trn_list), args.train_size)

            for idx in xrange(0, batch_idxs, self.batch_size):
            # for idx in xrange(0, 1):
                # TODO: Modify this
                img_A = []
                img_B = []
                labels = []
                beta = []
                if (idx + self.batch_size) > batch_idxs:
                    continue
                for img in range(self.batch_size):
                    
                    Data = load_data4FCN_CV(self,
                                            data_trn_list,
                                            sample_index=idx + img,
                                            labels2new_labels=labels2new_labels)
                    img_A.append(Data[0])
                    img_B.append(Data[1])
                    labels.append(Data[2])
                    beta.append(Data[3])
                if np.sum(beta) < 2:
                    continue
                batch_images = np.concatenate((np.array(img_A).reshape(self.batch_size, self.image_size, self.image_size, self.input_c_dim),
                                               np.array(img_B).reshape(self.batch_size, self.image_size, self.image_size, self.output_c_dim)),
                                              axis=3)
                labels = np.array(labels).reshape(self.batch_size, self.image_size, self.image_size)
                # Update FCN network
                _, fcn_loss = self.sess.run([fcn_optim, self.fcn_loss_real],
                                               feed_dict={self.real_data: batch_images, self.labels: labels, self.dropout: True})
                loss.append(fcn_loss)
                
            print ("fcn_ lossl -->", np.mean(loss))

            self.save(args.checkpoint_dir, counter)
            counter += 1
            # Validation generated samples !!!
            val_loss = validate_FCN_CV_batchsize2(self, data_test_list, labels2new_labels, real=True)
            Val_loss.append(val_loss)
            print("Training loss -->", np.mean(loss))
            print("Validation loss -->", val_loss)
            Trn_loss.append(np.mean(loss))
            errC_h = [Trn_loss, Val_loss]
            legends = ['trn loss', 'val loss']
            self.plot_loss(errC_h, legends, fig10)
            print("Epoch --->", epoch)
        # time.sleep(60*60*24*8)

        lossG_g = []
        lossG_L1 = []
        lossG = []
        lossD = []
        Val_loss = []
        Trn_loss = []
        fig10 = plt.figure('losses fcn')
        print("Training process begings ....")
        for epoch in xrange(args.epoch):
        # for epoch in xrange(1):
            loss = []
            G_gloss = []
            G_l1loss = []
            errorD = []
            errorG = []


            np.random.shuffle(data_trn_list)
            batch_idxs = min(len(data_trn_list), args.train_size)
            for idx in xrange(0, batch_idxs, self.batch_size):
            # for idx in xrange(0, 1):
                # TODO: Modify this
                img_A = []
                img_B = []
                labels = []
                beta = []
                if (idx + self.batch_size) > batch_idxs:
                    continue
                for img in range(self.batch_size):
                    
                    Data = load_data4FCN_CV(self,
                                            data_trn_list,
                                            sample_index=idx + img,
                                            labels2new_labels=labels2new_labels)
                    img_A.append(Data[0])
                    img_B.append(Data[1])
                    labels.append(Data[2])
                    beta.append(Data[3])
                if np.sum(beta) < 2:
                    continue
                beta = 1
                batch_images = np.concatenate((np.array(img_A).reshape(self.batch_size, self.image_size, self.image_size, self.input_c_dim),
                                               np.array(img_B).reshape(self.batch_size, self.image_size, self.image_size, self.output_c_dim)),
                                              axis=3)
                labels = np.array(labels).reshape(self.batch_size, self.image_size, self.image_size)            

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.real_data: batch_images, self.dropout: False})
                self.writer.add_summary(summary_str, counter)
                # print ("d_loss -->", d_loss)

                # Update G network
                _ = self.sess.run([g_optim],
                                  feed_dict={self.real_data: batch_images, self.labels: labels, self.dropout: False, self.beta: beta})
                # self.writer.add_summary(summary_str, counter)
                # print ("g_loss -->", g_loss)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _ = self.sess.run([g_optim],
                                  feed_dict={self.real_data: batch_images, self.labels: labels, self.dropout: False, self.beta: beta})
                # self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images, self.dropout: False})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images, self.dropout: False})
                # errG = self.g_loss.eval({self.real_data: batch_images, self.labels_classifier: g_trn_labels, self.dropout_keep_prob: 1.0})

                g_loss, g_lossl1, fcn_loss = self.sess.run([self.g_loss0, self.g_lossl1, self.fcn_loss_fake],
                                                feed_dict={self.real_data: batch_images, self.labels: labels, self.dropout: False, self.beta: beta})

                G_gloss.append(g_loss)
                G_l1loss.append(g_lossl1)
                loss.append(fcn_loss)
                
                errorD.append(errD_fake + errD_real)
                errorG.append(g_loss + g_lossl1 + fcn_loss)

                if np.mod(idx + 1, 99) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (counter, idx, batch_idxs,
                            time.time() - start_time, np.mean(errorD), np.mean(errorG)))
                    print ("gloss0 -->", np.mean(G_gloss), "glossl1 -->", np.mean(G_l1loss))

            tst_opt_generated = self.sess.run(self.fake_B_sample,
                                              feed_dict={self.real_A: tst_img_A})
            print(np.shape(tst_opt_generated))
            tst_opt_generated = tst_opt_generated[0].reshape(256, 256, self.n_features)
            fig = self.plot_patch(tst_opt_generated, n_fig="Generated: test")
            fig.savefig(sample_dir + '/sample_tst_' + str(counter) + '_.png', dpi=300)

            trn_opt_generated = self.sess.run(self.fake_B_sample,
                                              feed_dict={self.real_A: trn_img_A})
            trn_opt_generated = trn_opt_generated[0].reshape(256, 256, self.n_features)
            fig = self.plot_patch(trn_opt_generated, n_fig="Generated: train")
            fig.savefig(sample_dir + '/sample_trn_' + str(counter) + '_.png', dpi=300)

            print("-----------------------------------------------------")

            self.save(args.checkpoint_dir, counter)
            counter += 1
            # Save statistic by epoch
            lossG.append(np.mean(errorG))
            lossG_g.append(np.mean(G_gloss))
            lossG_L1.append(np.mean(G_l1loss))
            np.save(sample_dir + '/lossG', lossG)
            np.save(sample_dir + '/lossG_g', lossG_g)
            np.save(sample_dir + '/lossG_L1', lossG_L1)

            # Validation generated samples !!!
            val_loss = validate_FCN_CV_batchsize2(self, data_test_list, labels2new_labels)
            Val_loss.append(val_loss)
            print("Training loss -->", np.mean(loss))
            print("Validation loss -->", val_loss)
            Trn_loss.append(np.mean(loss))
            errC_h = [Trn_loss, Val_loss]
            legends = ['trn loss', 'val loss']
            self.plot_loss(errC_h, legends, fig10)
    # Fine tuning
    def fine_tuning(self, args):
        """Optimize fcn"""
        # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                        scope="hidden[34]|outputs")
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        fcn_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                            .minimize(self.fcn_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        labels_path = '/mnt/Data/DataBases/RS/SAR/Campo Verde/Labels_uint8/10_May_2016.tif'
        labels2new_labels, new_labels2labels = labels_look_table(labels_path)

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        sample_dir_root = '/home/jose/Templates/Pix2Pix/pix2pix-tensorflow_jose/sample/'
        sample_dir = os.path.join(sample_dir_root, self.dataset_name)

        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        # Cambiar para los datos de Campo Verde
        datasets_root = '/mnt/Data/Pix2Pix_datasets/Semi_Exp/' 
        dataset_name = '05may2016_C01_synthesize_semisupervised_multitemporal/'
        data_trn_list = glob.glob(datasets_root + dataset_name + 'Training/*.npy')
        data_val_list = glob.glob(datasets_root + dataset_name + 'Training/*.npy')
        data_test_list= glob.glob(datasets_root + dataset_name + 'Testing/*.npy')

        data_Dic = np.load(data_test_list[6]).item()
        tst_labels = np.array(data_Dic['labels'])
        tst_img_A = np.array(data_Dic['img_A']).astype('float32').reshape(1, 256, 256, self.input_c_dim)
        tst_img_B = np.array(data_Dic['img_B']).astype('float32')
        fig = self.plot_patch(tst_img_B, n_fig="Testing Patch")
        fig.savefig(sample_dir + '/sample_original_tst.png', dpi=300)

        data_Dic = np.load(data_trn_list[6]).item()
        trn_labels = np.array(data_Dic['labels'])
        trn_img_A = np.array(data_Dic['img_A']).astype('float32').reshape(1, 256, 256, self.input_c_dim)
        trn_img_B = np.array(data_Dic['img_B']).astype('float32')
        fig = self.plot_patch(trn_img_B, n_fig="Training Patch")
        fig.savefig(sample_dir + '/sample_original_trn.png', dpi=300)

        plt.figure("Testing Labels")
        plt.imshow(tst_labels)
        plt.show(block=False)
        plt.figure("Training Labels")
        plt.imshow(trn_labels)
        plt.show(block=False)
        plt.pause(0.5)

        Val_loss = []
        Trn_loss = []
        fig5 = plt.figure('losses fcn')
        for epoch in xrange(args.epoch):

            np.random.shuffle(data_trn_list)
            batch_idxs = min(len(data_trn_list), args.train_size) // self.batch_size

            loss = []
            tst_opt_generated = self.sess.run(self.fake_B_sample,
                                              feed_dict={self.real_A: tst_img_A})
            print(np.shape(tst_opt_generated))
            tst_opt_generated = tst_opt_generated.reshape(256, 256, self.n_features)
            fig = self.plot_patch(tst_opt_generated, n_fig="Generated: test")

            for idx in xrange(0, batch_idxs):
            # for idx in xrange(0, 10):
                # TODO: Modify this
                img_A, img_B, labels, beta = load_data4FCN_CV(self,
                                                              data_trn_list,
                                                              sample_index=idx,
                                                              labels2new_labels=labels2new_labels)

                if beta == 0:
                    continue

                batch_images = np.concatenate((img_A.reshape(1, 256, 256, self.input_c_dim),
                                               img_B.reshape(1, 256, 256, self.output_c_dim)),
                                              axis=3)
                # labels = np.random.randint(2, size=(256, 256))
                # Update G network
                # print(np.unique(labels))
                _, fcn_loss = self.sess.run([fcn_optim, self.fcn_loss],
                                           feed_dict={self.real_data: batch_images, self.labels: labels})
                # print (fcn_loss)
                loss.append(fcn_loss)
            print("Loss epoch --->", np.mean(loss))
            print("-----------------------------------------------------")
            # Validation generated samples !!!
            val_loss = validate_FCN_CV(self, data_test_list, labels2new_labels)
            Val_loss.append(val_loss)
            print("Validation loss -->", val_loss)
            Trn_loss.append(np.mean(loss))
            errC_h = [Trn_loss, Val_loss]
            legends = ['trn loss', 'val loss']
            self.plot_loss(errC_h, legends, fig5)
            # plt.savefig(sample_dir + 'loss_classifier_generated_validation.png', dpi=600)


    def plot_loss(self, data, legends, fig):
        # n_graph = len(legends)
        colors = ['r-', 'b-', 'g-', 'y-']
        sh = np.shape(data)
        n_dim = np.shape(sh)[0]
        if n_dim < 1:
            return 0
        ax = fig.add_subplot(111)
        ax.legend(legends, loc='upper right', fontsize=14)
        x = np.arange(len(data[0]))
        # print (x)
        for i in range(n_dim):
            y = data[i]       
            line, = ax.plot(x, y, colors[i])
            # fig.draw()
            # line, = ax.plot(x, y[1], 'b-')
            fig.show()
            plt.pause(0.001)
        return plt

    def plot_patch(self, patch, n_fig):
        im =  inverse_transform(patch[:, :, [2, 1, 0]])
        im[:, :, 0] = exposure.equalize_adapthist(im[:, :, 0], clip_limit=0.02)
        im[:, :, 1] = exposure.equalize_adapthist(im[:, :, 1], clip_limit=0.02)
        im[:, :, 2] = exposure.equalize_adapthist(im[:, :, 2], clip_limit=0.02)
        plt.figure(n_fig)
        plt.imshow(im)
        plt.show(block=False)
        plt.pause(0.5)
        return plt

    
    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h0 = tf.layers.dropout(h0, 0.1, training=self.dropout)
            # h0 = tf.nn.dropout(h0, 0.75)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h1 = tf.layers.dropout(h1, 0.1, training=self.dropout)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h2 = tf.layers.dropout(h2, 0.1, training=self.dropout)
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h3 = tf.layers.dropout(h3, 0.1, training=self.dropout)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            e1 = conv2dlayer(image, self.gf_dim, name='g_e1_conv', trainable=self.isTrain) # 64x2x5x5+64 = 3,264
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2dlayer(lrelu(e1), self.gf_dim*2, name='g_e2_conv', trainable=self.isTrain)) # (2x64)x64x5x5 + (2x64) = 204,928
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2dlayer(lrelu(e2), self.gf_dim*4, name='g_e3_conv', trainable=self.isTrain)) # (4x64)x(2x64)x5x5 + (4x64) = 819,456
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2dlayer(lrelu(e3), self.gf_dim*8, name='g_e4_conv', trainable=self.isTrain)) # (8x64)x(4x64)x5x5 + (8x64) = 3,277,312
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2dlayer(lrelu(e4), self.gf_dim*8, name='g_e5_conv', trainable=self.isTrain)) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2dlayer(lrelu(e5), self.gf_dim*8, name='g_e6_conv', trainable=self.isTrain)) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2dlayer(lrelu(e6), self.gf_dim*8, name='g_e7_conv', trainable=self.isTrain)) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2dlayer(lrelu(e7), self.gf_dim*8, name='g_e8_conv', trainable=self.isTrain)) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1= deconv2dlayer(tf.nn.relu(e8), self.gf_dim*8, name='g_d1', trainable=self.isTrain) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2 = deconv2dlayer(tf.nn.relu(d1), self.gf_dim*8, name='g_d2', trainable=self.isTrain) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3= deconv2dlayer(tf.nn.relu(d2), self.gf_dim*8, name='g_d3', trainable=self.isTrain) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4 = deconv2dlayer(tf.nn.relu(d3), self.gf_dim*8, name='g_d4', trainable=self.isTrain) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5 = deconv2dlayer(tf.nn.relu(d4), self.gf_dim*4, name='g_d5', trainable=self.isTrain) # (2*4x64)x(8x64)x5x5 + (2*4x64) = 6,554,112
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6 = deconv2dlayer(tf.nn.relu(d5), self.gf_dim*2, name='g_d6', trainable=self.isTrain) # (2*2x64)x(4x64)x5x5 + (2*2x64) = 1,638,656
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7 = deconv2dlayer(tf.nn.relu(d6), self.gf_dim, name='g_d7', trainable=self.isTrain) # (2*1x64)x(2x64)x5x5 + (2*1x64) = 409,728
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8 = deconv2dlayer(tf.nn.relu(d7), self.output_c_dim, name='g_d8', trainable=self.isTrain) # (1*1x64)x(1x7)x5x5 + (1*1x7) = 11,207
            # d8 is (256 x 256 x output_c_dim)

            self.g_output = tf.nn.tanh(self.d8)

            return self.g_output

    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            e1 = conv2dlayer(image, self.gf_dim, name='g_e1_conv', trainable=self.isTrain) # 64x2x5x5+64 = 3,264
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2dlayer(lrelu(e1), self.gf_dim*2, name='g_e2_conv', trainable=self.isTrain)) # (2x64)x64x5x5 + (2x64) = 204,928
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2dlayer(lrelu(e2), self.gf_dim*4, name='g_e3_conv', trainable=self.isTrain)) # (4x64)x(2x64)x5x5 + (4x64) = 819,456
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2dlayer(lrelu(e3), self.gf_dim*8, name='g_e4_conv', trainable=self.isTrain)) # (8x64)x(4x64)x5x5 + (8x64) = 3,277,312
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2dlayer(lrelu(e4), self.gf_dim*8, name='g_e5_conv', trainable=self.isTrain)) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2dlayer(lrelu(e5), self.gf_dim*8, name='g_e6_conv', trainable=self.isTrain)) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2dlayer(lrelu(e6), self.gf_dim*8, name='g_e7_conv', trainable=self.isTrain)) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2dlayer(lrelu(e7), self.gf_dim*8, name='g_e8_conv', trainable=self.isTrain)) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1= deconv2dlayer(tf.nn.relu(e8), self.gf_dim*8, name='g_d1', trainable=self.isTrain) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2 = deconv2dlayer(tf.nn.relu(d1), self.gf_dim*8, name='g_d2', trainable=self.isTrain) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3= deconv2dlayer(tf.nn.relu(d2), self.gf_dim*8, name='g_d3', trainable=self.isTrain) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4 = deconv2dlayer(tf.nn.relu(d3), self.gf_dim*8, name='g_d4', trainable=self.isTrain) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5 = deconv2dlayer(tf.nn.relu(d4), self.gf_dim*4, name='g_d5', trainable=self.isTrain) # (2*4x64)x(8x64)x5x5 + (2*4x64) = 6,554,112
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6 = deconv2dlayer(tf.nn.relu(d5), self.gf_dim*2, name='g_d6', trainable=self.isTrain) # (2*2x64)x(4x64)x5x5 + (2*2x64) = 1,638,656
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7 = deconv2dlayer(tf.nn.relu(d6), self.gf_dim, name='g_d7', trainable=self.isTrain) # (2*1x64)x(2x64)x5x5 + (2*1x64) = 409,728
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8 = deconv2dlayer(tf.nn.relu(d7), self.output_c_dim, name='g_d8', trainable=self.isTrain) # (1*1x64)x(1x7)x5x5 + (1*1x7) = 11,207
            # d8 is (256 x 256 x output_c_dim)

            self.g_output = tf.nn.tanh(self.d8)

            return self.g_output


    def classifier_fcn(self, image, reuse=False):
        with tf.variable_scope("classifier_fcn") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            e1 = conv2dlayer(image, self.gf_dim, name='fcn_e1_conv')
            e2 = self.fcn_bn_e2(conv2dlayer(lrelu(e1), self.gf_dim*2, name='fcn_e2_conv'))
            e3 = self.fcn_bn_e3(conv2dlayer(lrelu(e2), self.gf_dim*4, name='fcn_e3_conv'))

            self.d1= deconv2dlayer(tf.nn.relu(e3), self.gf_dim*2, name='fcn_d1')
            d1 = tf.layers.dropout(self.fcn_bn_d1(self.d1), 0.5, training=self.dropout)
            d1 = tf.concat([d1, e2], 3)

            self.d2 = deconv2dlayer(tf.nn.relu(d1), self.gf_dim, name='fcn_d2')
            d2 = tf.layers.dropout(self.fcn_bn_d2(self.d2), 0.5, training=self.dropout)
            d2 = tf.concat([d2, e1], 3)

            self.class_map = deconv2dlayer(tf.nn.relu(d2), self.n_classes, name='fcn_class_map')

            return self.class_map


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
            # sar_t0: scaler_name='sar_may2016_scaler.pkl'
            # opt_t0: scaler_name='opt_may2016_scaler.pkl'
            # sar_t1: scaler_name='sar_may2017_scaler.pkl'
            # opt_t1: scaler_name='opt_may2017_scaler.pkl'
            # img_A = np.concatenate((sar_t0, sar_t1, opt_t1), axis=2)
            scaler_sar_t0 = joblib.load("sar_may2016_scaler.pkl")
            scaler_sar_t1 = joblib.load("sar_may2017_scaler.pkl")
            scaler_opt_t1 = joblib.load("opt_may2017_scaler.pkl")
            print 'Case A ...'
            print 'generating image for_' + args.dataset_name
            sar_img_name_t0 = '10_08May_2016.npy'
            sar_img_name_t1 = '20170520.npy'
            sar_path_t0=self.sar_root_patch + sar_img_name_t0
            sar_path_t1=self.sar_root_patch + sar_img_name_t1
            sar_t0 = np.load(sar_path_t0)
            sar_t1 = np.load(sar_path_t1)
            sar_t0 = resampler(sar_t0, 'float32')
            sar_t1 = resampler(sar_t1, 'float32')
            sar_t0[sar_t0 > 1.0] = 1.0
            sar_t1[sar_t1 > 1.0] = 1.0
            num_rows, num_cols, num_bands = sar_t0.shape
            print('sar_t0.shape --->', sar_t0.shape)
            sar_t0 = sar_t0.reshape(num_rows * num_cols, num_bands)
            sar_t1 = sar_t1.reshape(num_rows * num_cols, num_bands)
            sar_t0 = np.float32(scaler_sar_t0.transform(sar_t0))
            sar_t1 = np.float32(scaler_sar_t1.transform(sar_t1))
            sar_t0 = sar_t0.reshape(num_rows, num_cols, num_bands)
            sar_t1 = sar_t1.reshape(num_rows, num_cols, num_bands)

            opt_name_t1 = '20170524/'
            opt_path_t1=self.opt_root_patch + opt_name_t1
            opt_t1, _ = load_landsat(opt_path_t1)
            opt_t1[np.isnan(opt_t1)] = 0.0
            print("opt_t1 -->", opt_t1.shape)
            opt_t1 = opt_t1.reshape(num_rows * num_cols, self.output_c_dim)
            opt_t1 = np.float32(scaler_opt_t1.transform(opt_t1))
            opt_t1 = opt_t1.reshape(num_rows, num_cols, self.output_c_dim)
            img_A = np.concatenate((sar_t0, sar_t1, opt_t1), axis=2)
            img_A = img_A.reshape(1, num_rows, num_cols, self.input_c_dim)
            fake_opt = np.zeros((num_rows, num_cols, self.output_c_dim),
                                dtype='float32')
            
            s = 64
            stride = self.image_size-2*s
            for row in range(0, num_rows, stride):
                for col in range(0, num_cols, stride):
                    if (row+self.image_size <= num_rows) and (col+self.image_size <= num_cols):

                        print row + s, row + self.image_size - s
                        sample_image = img_A[:, row:row+self.image_size, col:col+self.image_size]
                        sample = self.sess.run(self.fake_B_sample,
                                               feed_dict={self.real_A: sample_image}
                                               )
                        print sample.shape
                        fake_opt[row+s:row+self.image_size-s, col+s:col+self.image_size-s] = sample[0, s:self.image_size-s, s:self.image_size-s]
                    elif col+self.image_size <= num_cols:
                        sample_image = img_A[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
                        print(sample_image.shape)
                        sample = self.sess.run(self.fake_B_sample,
                                               feed_dict={self.real_A: sample_image}
                                               )
                        print sample.shape
                        fake_opt[row+s:num_rows, col+s:col+self.image_size-s] = sample[0, self.image_size-num_rows+row+s:self.image_size, s:self.image_size-s]
                    elif row+self.image_size <= num_rows:
                        print col
                        sample_image = img_A[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
                        sample = self.sess.run(self.fake_B_sample,
                                               feed_dict={self.real_A: sample_image}
                                               )
                        fake_opt[row+s:row+self.image_size-s, col+s:num_cols] = sample[0, s:self.image_size-s, self.image_size-num_cols+col+s:self.image_size]

            np.save(self.dataset_name + '_fake_opt', fake_opt)
            # MONOTEMPORAL
            # sar_img_name = '10_08May_2016.npy'
            # sar_path=self.sar_root_patch + sar_img_name
            # sar = np.load(sar_path)
            # sar = resampler(sar, 'float32')
            # sar[sar > 1.0] = 1.0
            # num_rows, num_cols, num_bands = sar.shape
            # sar = sar.reshape(num_rows * num_cols, num_bands)
            # scaler = joblib.load("sar_05may2016_scaler.pkl")
            # sar = np.float32(scaler.transform(sar))
            # img_source = sar.reshape(1, num_rows, num_cols, num_bands)
            # fake_opt = np.zeros((num_rows, num_cols, self.output_c_dim),
            #                     dtype='float32')
            # s = 64
            # stride = self.image_size-2*s
            # for row in range(0, num_rows, stride):
            #     for col in range(0, num_cols, stride):
            #         if (row+self.image_size <= num_rows) and (col+self.image_size <= num_cols):

            #             print row + s, row + self.image_size - s
            #             sample_image = img_source[:, row:row+self.image_size, col:col+self.image_size]
            #             sample = self.sess.run(self.fake_B_sample,
            #                                    feed_dict={self.real_A: sample_image}
            #                                    )
            #             print sample.shape
            #             fake_opt[row+s:row+self.image_size-s, col+s:col+self.image_size-s] = sample[0, s:self.image_size-s, s:self.image_size-s]
            #         elif col+self.image_size <= num_cols:
            #             sample_image = img_source[:, num_rows-self.image_size:num_rows, col:col+self.image_size]
            #             print(sample_image.shape)
            #             sample = self.sess.run(self.fake_B_sample,
            #                                    feed_dict={self.real_A: sample_image}
            #                                    )
            #             print sample.shape
            #             fake_opt[row+s:num_rows, col+s:col+self.image_size-s] = sample[0, self.image_size-num_rows+row+s:self.image_size, s:self.image_size-s]
            #         elif row+self.image_size <= num_rows:
            #             print col
            #             sample_image = img_source[:, row:row+self.image_size, num_cols-self.image_size:num_cols]
            #             sample = self.sess.run(self.fake_B_sample,
            #                                    feed_dict={self.real_A: sample_image}
            #                                    )
            #             fake_opt[row+s:row+self.image_size-s, col+s:num_cols] = sample[0, s:self.image_size-s, self.image_size-num_cols+col+s:self.image_size]

            # np.save(self.dataset_name + '_fake_opt', fake_opt)

    def create_dataset(self, args):
        if '05may2016' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            # sar_img_name = '10_08May_2016.npy'
            # opt_img_name = '20160505/'
            # print sar_img_name, opt_img_name
            # create_dataset_4_classifier(
            #     ksize=256,
            #     dataset=self.dataset_name,
            #     mask_path=None,
            #     sar_path=self.sar_root_patch + sar_img_name,
            #     opt_path=self.opt_root_patch + opt_img_name
            # )
            sar_t0 = '10_08May_2016.npy'
            opt_t0 = '20160505/'
            sar_t1 = '20170520.npy'
            opt_t1 = '20170524/'
            # print sar_img_name, opt_img_name
            create_dataset_4_classifier_multitemporal(
                ksize=256,
                dataset=self.dataset_name,
                mask_path=None,
                sar_path_t0=self.sar_root_patch + sar_t0,
                opt_path_t0=self.opt_root_patch + opt_t0,
                sar_path_t1=self.sar_root_patch + sar_t1,
                opt_path_t1=self.opt_root_patch + opt_t1,
            )
        elif 'quemadas_ap2_case_A' in args.dataset_name:
            print 'creating dataset for_' + args.dataset_name
            create_dataset_case_A(
                ksize=256,
                dataset=self.dataset_name,
                mask_path=None,
                sar_path='/mnt/Data/DataBases/RS/Quemadas/AP2_Acre/Sentinel1/20160909/new_20160909.npy',
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
            return 0
        print 'creating dataset for_' + args.dataset_name
