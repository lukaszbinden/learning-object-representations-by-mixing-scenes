import os
import time

import tensorflow as tf

from ops_alex import *
from utils import *
from input_pipeline_rendered_data_sprites import get_pipeline_training_from_dump

import math
import numpy as np
import scipy.io as sio


class DCGAN(object):

    def __init__(self, sess,
                 batch_size=256, sample_size = 64, image_shape=[256, 256, 3],
                 y_dim=None, z_dim=0, gf_dim=128, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1, is_train=True):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.model_name = "DCGAN.model"
        self.sess = sess
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.image_shape = image_shape
        self.image_size = image_shape[0]

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.z = None

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        """ df_dim: Dimension of discrim filters in first conv layer. [64] """

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        """ c_dim: Dimension of image color. [3] """
        self.cg_dim = cg_dim


        self.d_bn1 = batch_norm(is_train, name='d_bn1')
        self.d_bn2 = batch_norm(is_train, name='d_bn2')
        self.d_bn3 = batch_norm(is_train, name='d_bn3')
        self.d_bn4 = batch_norm(is_train, name='d_bn4')

        self.c_bn1 = batch_norm(is_train, name='c_bn1')
        self.c_bn2 = batch_norm(is_train, name='c_bn2')
        self.c_bn3 = batch_norm(is_train, name='c_bn3')
        self.c_bn4 = batch_norm(is_train, name='c_bn4')
        self.c_bn5 = batch_norm(is_train, name='c_bn5')

        # TODO not used?
        self.g_s_bn5 = batch_norm(is_train,convolutional=False, name='g_s_bn5')

        self.build_model()


    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.abstract_size = self.sample_size // 2 ** 4

        _, _, images = get_pipeline_training_from_dump('data_example.tfrecords',
                                                                 self.batch_size*3,
                                                                 1000, image_size=60,resize_size=60,
                                                                 img_channels=self.c_dim)

        _, _, test_images1 = get_pipeline_training_from_dump('data_example.tfrecords',
                                                                 self.batch_size*2,
                                                                 10000000, image_size=60,resize_size=60,
                                                                 img_channels=self.c_dim)

        self.images_x1 = images[0:self.batch_size, :, :, :]
        """ images_x1: tensor of images (64, 60, 60, 3) """
        self.images_x2 = images[self.batch_size:self.batch_size * 2, :, :, :]

        self.third_image = images[self.batch_size*2:self.batch_size*3,:,:,:]

        self.test_images1 = test_images1[0:self.batch_size,:,:,:]
        self.test_images2 = test_images1[self.batch_size:self.batch_size*2,:,:,:]

        self.chunk_num = 8
        """ number of chunks: 8 """
        self.chunk_size = 64
        """ size per chunk: 64 """
        self.feature_size = self.chunk_size*self.chunk_num

        with tf.variable_scope('generator') as scope_generator:
            # Enc for x1
            self.f_1 = self.encoder(self.images_x1)
            """ feature rep f_1 as 2D vector (batch_size, feature_size) -> (64, 512) """
            # Dec for x1 -> x1_hat
            self.images_x1_hat = self.generator(self.f_1)
            """ batch of reconstructed images with size (64, 60, 60, 3) """
            # Cls
            # TODO this line seems useless
            _ = self.classifier(self.images_x1_hat, self.images_x1_hat, self.images_x1_hat)

            # to share the weights between the Encoders
            scope_generator.reuse_variables()
            # Enc for x2
            self.f_2 = self.encoder(self.images_x2)
            """ feature rep f_2 as 2D vector (batch_size, feature_size) -> (64, 512) """
            # Dec for x2
            self.images_x2_hat = self.generator(self.f_2)

            # for the mask e.g. [1 1 0 0 1 1 0 0], of shape (8,)
            mask = tf.random_uniform(shape=[self.chunk_num],minval=0,maxval=2,dtype=tf.int32)
            # each chunk is initialized with 1's (64,64)
            a_chunk = tf.ones((self.batch_size,self.chunk_size),dtype=tf.int32)
            # TODO this line seems useless
            a_fea = tf.ones_like(self.f_1, dtype=tf.int32)

            # chunk stuff: i -> chunk-id
            i=0
            f_1_chunk = self.f_1[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            f_2_chunk = self.f_2[:, i * self.chunk_size:(i + 1) * self.chunk_size]

            # all params with R -> for 2nd image x2
            self.f_1_2 = tf.where(tf.equal(mask[i] * a_chunk, 0), f_1_chunk, f_2_chunk)
            """ f_1_2: used to be f_1_mix """
            self.f_2_mix = tf.where(tf.equal(mask[i] * a_chunk, 1), f_1_chunk, f_2_chunk)

            # mix the feature (cf step 2)
            for i in range(1, self.chunk_num): # for each chunk
                f_1_chunk = self.f_1[:, i * self.chunk_size:(i + 1) * self.chunk_size]
                f_2_chunk = self.f_2[:, i * self.chunk_size:(i + 1) * self.chunk_size]
                self.f_chunk_selected = tf.where(tf.equal(mask[i] * a_chunk, 0), f_1_chunk, f_2_chunk)
                self.f_1_2 = tf.concat(axis=1, values=[self.f_1_2, self.f_chunk_selected])

                self.f_chunk_selected = tf.where(tf.equal(mask[i] * a_chunk, 1), f_1_chunk, f_2_chunk)
                # TODO f_2_mix -> seems unused
                self.f_2_mix = tf.concat(axis=1, values=[self.f_2_mix, self.f_chunk_selected])


            # TODO why is this stored as instance variable?
            self.k = mask
            """ mask is e.g. [1 1 0 0 1 1 0 0] """
            # TODO: k0 not used?
            self.k0 = mask[0]

            # Dec x3
            self.images_x3 = self.generator(self.f_1_2)
            # Cls (input x1, x2, x3)
            self.cls = self.classifier(self.images_x1, self.images_x2, self.images_x3)
            """ cls is of size (64, 8) """

            # cf original mask
            self.mask_batchsize = tf.cast(tf.ones((self.batch_size, self.chunk_num), dtype=tf.int32) * mask, tf.float32)
            """ kfc: mask (8,) scaled to batch_size, of shape (64, 8) """

            # rep_mix = f3 (Enc for f3)
            self.f_3 = self.encoder(self.images_x3)
            """ f_3: feature rep as 2D vector (batch_size, feature_size) -> (64, 512) """

            # from f3 to f31/f32 START
            i = 0
            f_3_chunk = self.f_3[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            f_1_chunk = self.f_1[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            f_2_chunk = self.f_2[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            self.f_3_1 = tf.where(tf.equal(mask[i] * a_chunk, 0), f_3_chunk, f_1_chunk)
            """ f_3_1: used to be rep_re; of shape (64, 512) """
            self.f_3_2 = tf.where(tf.equal(mask[i] * a_chunk, 1), f_3_chunk, f_2_chunk)
            """ f_3_2: used to be repR_re """

            for i in range(1, self.chunk_num):
                f_3_chunk = self.f_3[:, i * self.chunk_size:(i + 1) * self.chunk_size]
                f_1_chunk = self.f_1[:, i * self.chunk_size:(i + 1) * self.chunk_size]
                f_2_chunk = self.f_2[:, i * self.chunk_size:(i + 1) * self.chunk_size]
                self.f_chunk_selected = tf.where(tf.equal(mask[i] * a_chunk, 0), f_3_chunk, f_1_chunk)
                self.f_3_1 = tf.concat(axis=1, values=[self.f_3_1, self.f_chunk_selected])
                self.f_chunk_selected = tf.where(tf.equal(mask[i] * a_chunk, 1), f_3_chunk, f_2_chunk)
                self.f_3_2 = tf.concat(axis=1, values=[self.f_3_2, self.f_chunk_selected])
            # from f3 to f31/f32 END

            # from f31 Dec to x4
            self.images_x4 = self.generator(self.f_3_1)
            """ images_x4: batch of reconstructed images x4 with shape (64, 60, 60, 3) """
            # from f32 to x4' (check..)
            self.images_x4_hat = self.generator(self.f_3_2)

            scope_generator.reuse_variables()
            # for test only
            self.f_test_1 = self.encoder(self.test_images1)
            self.f_test_2 = self.encoder(self.test_images2)

            ####################################################
            # mix the features for the two test images_x1 START
            i = 0
            self.f_test_1_2 = self.f_test_2[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            for i in range(1, self.chunk_num):
                tmp = self.f_test_1[:, i * self.chunk_size:(i + 1) * self.chunk_size]
                self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
            self.D_mix_allchunk = self.generator(self.f_test_1_2, reuse=True)
            self.D_mix_allchunk_sup = self.D_mix_allchunk


            for i in range(1,self.chunk_num):
                self.f_test_1_2 = self.f_test_1[:, 0 * self.chunk_size:1 * self.chunk_size]
                for j in range(1,self.chunk_num):
                    if j==i:
                        tmp = self.f_test_2[:, j * self.chunk_size:(j + 1) * self.chunk_size]
                        self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
                    else:
                        tmp = self.f_test_1[:, j * self.chunk_size:(j + 1) * self.chunk_size]
                        self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
                tmp_mix = self.generator(self.f_test_1_2)
                self.D_mix_allchunk = tf.concat(axis=0,values=[self.D_mix_allchunk,tmp_mix])

            for i in range(1,self.chunk_num):
                self.f_test_1_2 = self.f_test_2[:, 0 * self.chunk_size:1 * self.chunk_size]
                for j in range(1,self.chunk_num):
                    if j<=i:
                        tmp = self.f_test_2[:, j * self.chunk_size:(j + 1) * self.chunk_size]
                        self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
                    else:
                        tmp = self.f_test_1[:, j * self.chunk_size:(j + 1) * self.chunk_size]
                        self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
                tmp_mix = self.generator(self.f_test_1_2)
                self.D_mix_allchunk_sup = tf.concat(axis=0,values=[self.D_mix_allchunk_sup,tmp_mix])
            # mix the features for the two test images_x1 END
            ####################################################

        with tf.variable_scope('classifier_loss'):
            # Cls loss; mask_batchsize here is GT, cls should predict correct mask..
            self.cls_loss = binary_cross_entropy_with_logits(self.mask_batchsize, self.cls)
            """ cf_loss: a scalar, of shape () """

        with tf.variable_scope('discriminator'):
            # Dsc for x1
            self.dsc_x1 = self.discriminator(self.images_x1)
            """ Dsc_x1: real/fake, of shape (64, 1) """
            # Dsc for x3
            self.dsc_x3 = self.discriminator(self.images_x3, reuse=True)
            """ Dsc_x3: real/fake, of shape (64, 1) """

        with tf.variable_scope('discriminator_loss'):
            # Dsc loss x1
            self.dsc_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_x1), self.dsc_x1)
            # Dsc loss x3
            self.dsc_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.dsc_x3), self.dsc_x3)
            self.dsc_loss = self.dsc_loss_real + self.dsc_loss_fake
            """ dsc_loss: a scalar, of shape () """

        with tf.variable_scope('generator_loss'):
            # D (fix Dsc you have loss for G) -> cf. Dec
            # images_x3 = Dec(f_1_2) = G(f_1_2); Dsc(images_x3) = dsc_x3
            # TODO rationale behind g_loss not clear yet
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_x3), self.dsc_x3)

        with tf.variable_scope('L2') as _:
            # Reconstruction loss L2 between x1 and x1' (to ensure autoencoder works properly)
            self.rec_loss_x1hat_x1 = tf.reduce_mean(tf.square(self.images_x1_hat - self.images_x1))
            """ rec_loss_x1hat_x1: a scalar, of shape () """
            # Reconstruction loss L2 between x2 and x2' (to ensure autoencoder works properly)
            self.rec_loss_x2hat_x2 = tf.reduce_mean(tf.square(self.images_x2_hat - self.images_x2))
            # L2 for x1 and x4
            self.rec_loss_x4_x1 = tf.reduce_mean(tf.square(self.images_x4 - self.images_x1))
            # L2 for x2 and x4'
            self.rec_loss_x4hat_x2 = tf.reduce_mean(tf.square(self.images_x4_hat - self.images_x2))

        # TODO what for?
        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()
        # Tf stuff (tell variables how to train..)
        self.d_vars = [var for var in t_vars if 'd_' in var.name] # discriminator
        self.g_vars = [var for var in t_vars if 'g_' in var.name] # encoder + generator/decoder
        self.g_s_vars = [var for var in t_vars if 'g_s' in var.name] # prob not used
        self.g_e_vars = [var for var in t_vars if 'g_en' in var.name] # prob not used
        self.c_vars = [var for var in t_vars if 'c_' in var.name] # classifier

        # save the weights
        self.saver = tf.train.Saver(self.d_vars + self.g_vars + self.c_vars + batch_norm.shadow_variables, max_to_keep=0)
        # END of build_model

    def train(self, config, run_string="???"):
        """Train DCGAN"""

        if config.continue_from_iteration:
            counter = config.continue_from_iteration
        else:
            counter = 0

        global_step = tf.Variable(counter, name='global_step', trainable=False)

        # Learning rate of generator is gradually decreasing.
        self.g_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        self.d_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        self.c_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        labmda = 0 # should be 0
        # g_loss = labmda*self.rec_loss+10*self.recR_loss +10*self.rec_mix_loss+1*self.g_loss+1*self.cf_loss

        # entire autoencoder loss (2 Enc and 2 Dec share weights) (between x1/x2 and x4)
        g_loss = 10 * self.rec_loss_x2hat_x2 + 10 * self.rec_loss_x4_x1 + 1 * self.g_loss + 1 * self.cls_loss
        # for autoencoder
        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=config.beta1) \
                          .minimize(g_loss, var_list=self.g_vars)
        # for classifier
        c_optim = tf.train.AdamOptimizer(learning_rate=self.c_learning_rate, beta1=config.beta1) \
                          .minimize(self.cls_loss, var_list=self.c_vars)
        # for Dsc
        d_optim = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=config.beta1) \
                          .minimize(self.dsc_loss, var_list=self.d_vars, global_step=global_step)

        # what you specify in the argument to control_dependencies is ensured to be evaluated before anything you define in the with block
        with tf.control_dependencies([g_optim]):
            g_optim = tf.group(self.bn_assigners) # TODO don't understand this...

        tf.global_variables_initializer().run()
        if config.continue_from:
            checkpoint_dir = os.path.join(os.path.dirname(config.checkpoint_dir), config.continue_from)
            print('Loading variables from ' + checkpoint_dir)
            self.load(checkpoint_dir, config.continue_from_iteration)

        start_time = time.time()

        # simple mechanism to coordinate the termination of a set of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        self.make_summary_ops()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.summary_dir, graph_def=self.sess.graph_def)

        try:
            # Training
            while not coord.should_stop():
                # Update D and G network
                tic = time.time()
                self.sess.run([g_optim])
                self.sess.run([c_optim])
                self.sess.run([d_optim])
                toc = time.time()
                counter += 1
                print(counter)
                duration = toc - tic

                if counter % 200 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, counter)

                if np.mod(counter, 4000) == 2:
                    # print out images every 4000 batches
                    images_x1,images_x2, images_x3, D_mix_allchunk,test_images1,test_images2,\
                    images_x1_hat,images_x2_hat,third_image,\
                    D_mix_allchunk_sup,images_x4,images_x4_hat, _, _ = \
                        self.sess.run([self.images_x1, self.images_x2, \
                             self.images_x3, self.D_mix_allchunk, self.test_images1, self.test_images2, \
                             self.images_x1_hat, self.images_x2_hat, self.third_image, \
                             self.D_mix_allchunk_sup, self.images_x4, self.images_x4_hat, \
                             self.D_mix_allchunk, self.D_mix_allchunk_sup])

                    grid_size = np.ceil(np.sqrt(self.batch_size))
                    grid = [grid_size, grid_size]
                    grid_celebA = [12, self.chunk_num+2]

                    save_images(images_x1,grid, os.path.join(config.summary_dir, '%s_train_images_x1.png' % counter))
                    save_images(images_x2, grid, os.path.join(config.summary_dir, '%s_train_images_x2.png' % counter))
                    save_images(images_x1_hat,grid, os.path.join(config.summary_dir, '%s_train_images_x1_hat.png' % counter))
                    save_images(images_x2_hat, grid, os.path.join(config.summary_dir, '%s_train_images_x2_hat.png' % counter))
                    save_images(images_x3, grid, os.path.join(config.summary_dir, '%s_train_images_x3.png' % counter))
                    save_images(images_x4, grid, os.path.join(config.summary_dir, '%s_train_images_x4.png' % counter))
                    save_images(images_x4_hat, grid, os.path.join(config.summary_dir, '%s_train_images_x4_hat.png' % counter))

                    save_images_multi(test_images1,test_images2,D_mix_allchunk, grid_celebA,self.batch_size, os.path.join(config.summary_dir, '%s_test1.png' % counter))
                    save_images_multi(test_images1,test_images2,D_mix_allchunk_sup, grid_celebA,self.batch_size, os.path.join(config.summary_dir, '%s_test_sup1.png' % counter))


                if np.mod(counter, 2000) == 0:
                    self.save(config.checkpoint_dir, counter)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)
        # END of train()


    def discriminator(self, image, keep_prob=0.5, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # cf. DCGAN impl https://github.com/carpedm20/DCGAN-tensorflow.git
        h0 = lrelu(self.d_bn1(conv2d(image, self.df_dim, name='d_1_h0_conv')))
        h1 = lrelu(self.d_bn2(conv2d(h0, self.df_dim*2, name='d_1_h1_conv')))
        h2 = lrelu(self.d_bn3(conv2d(h1, self.df_dim*4, name='d_1_h2_conv')))
        h3 = lrelu(self.d_bn4(conv2d(h2, self.df_dim*8,k_h=1, k_w=1, d_h=1, d_w=1, name='d_1_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_1_h3_lin')

        return tf.nn.sigmoid(h4)


    def classifier(self, image1, image2, image3, reuse=False):
        """From paper:
        For the classifier, we use AlexNet with batch normalization after each
        convolutional layer, but we do not use any dropout. The image inputs of
        the classifier are concatenated along the RGB channels.

        returns: a 1D matrix of size self.chunk_num
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        concated = tf.concat(axis=3, values=[image1, image2])
        concated = tf.concat(axis=3, values=[concated, image3])
        """ concated is of size (batch_size, 60, 60, 9) """

        conv1 = self.c_bn1(conv(concated, 96, 8,8,2,2, padding='VALID', name='c_3_s0_conv'))
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='c_3_mp0')

        conv2 = self.c_bn2(conv(pool1, 256, 5,5,1,1, groups=2, name='c_3_conv2'))
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='c_3_pool2')

        conv3 = self.c_bn3(conv(pool2, 384, 3, 3, 1, 1, name='c_3_conv3'))

        conv4 = self.c_bn4(conv(conv3, 384, 3, 3, 1, 1, groups=2, name='c_3_conv4'))

        conv5 = self.c_bn5(conv(conv4, 256, 3, 3, 1, 1, groups=2, name='c_3_conv5'))
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='c_3_pool5')

        fc6 = tf.nn.relu(linear(tf.reshape(pool5, [self.batch_size, -1]), 4096, 'c_3_fc6') )

        fc7 = tf.nn.relu(linear(tf.reshape(fc6, [self.batch_size, -1]), 4096, 'c_3_fc7') )

        self.fc8 = linear(tf.reshape(fc7, [self.batch_size, -1]), self.chunk_num, 'c_3_fc8')

        return tf.nn.sigmoid(self.fc8)


    def encoder(self, sketches_or_abstract_representations, reuse=False):
        """
        returns: 1D vector f1 with size=self.feature_size
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s0 = lrelu(instance_norm(conv2d(sketches_or_abstract_representations, self.df_dim, k_h=4, k_w=4, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, self.df_dim * 2, k_h=4, k_w=4, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, self.df_dim * 4, k_h=4, k_w=4, name='g_1_conv2')))
        s3 = lrelu(instance_norm(conv2d(s2, self.df_dim * 8, k_h=2, k_w=2, name='g_1_conv3')))
        s4 = lrelu(instance_norm(conv2d(s3, self.df_dim * 8, k_h=2, k_w=2, name='g_1_conv4')))
        used_abstract = lrelu((linear(tf.reshape(s4, [self.batch_size, -1]), self.feature_size, 'g_1_fc')))

        return used_abstract


    def generator(self, representations, reuse=False):
        """
        returns: batch of images with size 256x60x60x3
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshape = tf.reshape(representations,[self.batch_size, 1, 1, self.feature_size])
        h = deconv2d(reshape, [self.batch_size, 4, 4, self.gf_dim*4], k_h=4, k_w=4, d_h=1, d_w=1, padding='VALID', name='g_de_h')
        h = tf.nn.relu(h)

        h1 = deconv2d(h, [self.batch_size, 8, 8, self.gf_dim*4 ], name='g_h1')
        h1 = tf.nn.relu(instance_norm(h1))

        h2 = deconv2d(h1, [self.batch_size, 15, 15, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(instance_norm(h2))

        h3 = deconv2d(h2, [self.batch_size, 30, 30, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(instance_norm(h3))

        h4 = deconv2d(h3, [self.batch_size, 60, 60, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)


    def make_summary_ops(self):
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('classifier_loss', self.cls_loss)
        tf.summary.scalar('d_loss_fake', self.dsc_loss_fake)
        tf.summary.scalar('d_loss_real', self.dsc_loss_real)
        tf.summary.scalar('rec_loss', self.rec_loss_x1hat_x1)
        tf.summary.scalar('rec_mix_loss', self.rec_loss_x4_x1)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir, iteration=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and iteration:
            # Restores dump of given iteration
            ckpt_name = self.model_name + '-' + str(iteration)
        elif ckpt and ckpt.model_checkpoint_path:
            # Restores most recent dump
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

        ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
        print('Reading variables to be restored from ' + ckpt_file)
        self.saver.restore(self.sess, ckpt_file)
        return ckpt_name
