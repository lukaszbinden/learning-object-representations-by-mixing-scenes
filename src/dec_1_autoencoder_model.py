import os

from ops_alex import *
from utils_dcgan import *
from utils_common import *
from input_pipeline_rendered_data import get_pipeline_training_from_dump
from constants import *

import numpy as np


class DCGAN(object):

    def __init__(self, sess,
                 batch_size=256, sample_size=64, epochs=1000, image_shape=[256, 256, 3],
                 y_dim=None, z_dim=0, gf_dim=128, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1,
                 is_train=True):
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
        self.epochs = epochs

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

        self.build_model()


    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.abstract_size = self.sample_size // 2 ** 4

        _, _, images = get_pipeline_training_from_dump(dump_file='datasets/coco/2017_val/2017_val.tfrecords',
                                                                 batch_size=self.batch_size*3,
                                                                 epochs=self.epochs,
                                                                 image_size=300,resize_size=300,
                                                                 img_channels=self.c_dim)

        self.images_x1 = images[0:self.batch_size, :, :, :]
        """ images_x1: tensor of images (64, 60, 60, 3) """

        overlap = 9 # TODO hyperparameter
        # assert overlap, 'hyperparameter \'overlap\' is not an integer'
        image_size = 300
        slice_size = (image_size + 2 * overlap) / 3
        assert slice_size.is_integer(), 'hyperparameter \'overlap\' invalid: %d' % overlap
        slice_size = int(slice_size)
        slice_size_overlap = slice_size - overlap
        slice_size_overlap = int(slice_size_overlap)

        # (64, 24, 24, 3)
        self.tile_r1c1 = tf.image.crop_to_bounding_box(self.images_x1, 0, 0, slice_size, slice_size)
        self.tile_r1c2 = tf.image.crop_to_bounding_box(self.images_x1, 0, slice_size_overlap, slice_size, slice_size)
        self.tile_r1c3 = tf.image.crop_to_bounding_box(self.images_x1, 0, image_size - slice_size, slice_size, slice_size)
        self.tile_r2c1 = tf.image.crop_to_bounding_box(self.images_x1, slice_size_overlap, 0, slice_size, slice_size)
        self.tile_r2c2 = tf.image.crop_to_bounding_box(self.images_x1, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        self.tile_r2c3 = tf.image.crop_to_bounding_box(self.images_x1, slice_size_overlap, image_size - slice_size, slice_size, slice_size)
        self.tile_r3c1 = tf.image.crop_to_bounding_box(self.images_x1, image_size - slice_size, 0, slice_size, slice_size)
        self.tile_r3c2 = tf.image.crop_to_bounding_box(self.images_x1, image_size - slice_size, slice_size_overlap, slice_size, slice_size)
        self.tile_r3c3 = tf.image.crop_to_bounding_box(self.images_x1, image_size - slice_size, image_size - slice_size, slice_size, slice_size)


        self.chunk_num = 8
        """ number of chunks: 8 """
        self.chunk_size = 64
        """ size per chunk: 64 """
        self.feature_size = self.chunk_size*self.chunk_num

        with tf.variable_scope('generator') as scope_generator:
            # Enc for x1
            self.f_1 = self.encoder(self.tile_r1c1)
            # (64, 512)

            self.f_x1_composite = tf.zeros((self.batch_size, NUM_TILES * self.feature_size))
            # this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.decoder_image(self.f_x1_composite)

            # to share the weights between the Encoders
            scope_generator.reuse_variables()
            self.f_2 = self.encoder(self.tile_r1c2)
            self.f_3 = self.encoder(self.tile_r1c3)
            self.f_4 = self.encoder(self.tile_r2c1)
            self.f_5 = self.encoder(self.tile_r2c2)
            self.f_6 = self.encoder(self.tile_r2c3)
            self.f_7 = self.encoder(self.tile_r3c1)
            self.f_8 = self.encoder(self.tile_r3c2)
            self.f_9 = self.encoder(self.tile_r3c3)

            # build composite feature including all x1 tile features
            self.f_x1_composite = tf.concat([self.f_1, self.f_2, self.f_3, self.f_4, self.f_5, self.f_6, self.f_7, self.f_8, self.f_9], 1)
            # print(self.f_x1_all.get_shape())
            # Dec for x1 -> x1_hat
            self.images_x1_hat = self.decoder_image(self.f_x1_composite)
            # print(self.images_x1_hat)


        with tf.variable_scope('L2') as _:
            # Reconstruction loss L2 between x1 and x1' (to ensure autoencoder works properly)
            self.rec_loss_x1hat_x1 = tf.reduce_mean(tf.square(self.images_x1_hat - self.images_x1))
            """ rec_loss_x1hat_x1: a scalar, of shape () """

        t_vars = tf.trainable_variables()
        # Tf stuff (tell variables how to train..)
        self.gen_vars = [var for var in t_vars if 'g_' in var.name] # encoder + decoder (generator)

        # save the weights
        self.saver = tf.train.Saver(self.gen_vars + batch_norm.shadow_variables, max_to_keep=0)
        # END of build_model

    def train(self, params):
        """Train DCGAN"""

        if params.continue_from_iteration:
            counter = params.continue_from_iteration
        else:
            counter = 0

        global_step = tf.Variable(counter, name='global_step', trainable=False)

        # Learning rate of generator is gradually decreasing.
        self.g_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        g_loss_comp = self.rec_loss_x1hat_x1

        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=params.beta1) \
                          .minimize(g_loss_comp, var_list=self.gen_vars) # includes encoder + decoder weights


        tf.global_variables_initializer().run()
        if params.continue_from:
            ckpt_name = self.load(params, params.continue_from_iteration)
            counter = int(ckpt_name[ckpt_name.rfind('-')+1:])
            global_step.load(counter) # load new initial value into variable

        # simple mechanism to coordinate the termination of a set of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        self.make_summary_ops(g_loss_comp)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(params.summary_dir)
        summary_writer.add_graph(self.sess.graph)

        try:
            # Training
            while not coord.should_stop():
                # Update D and G network
                self.sess.run([g_optim])

                counter += 1
                print(str(counter))

                if counter % 50 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, counter)

                if np.mod(counter, 300) == 2:
                    images_x1, images_x1_hat = self.sess.run([self.images_x1, self.images_x1_hat])
                    grid_size = np.ceil(np.sqrt(self.batch_size))
                    grid = [grid_size, grid_size]
                    save_images(images_x1, grid, os.path.join(params.summary_dir, '%s_train_images_x1.png' % counter))
                    save_images(images_x1_hat, grid, os.path.join(params.summary_dir, '%s_train_images_x1_hat.png' % counter))

                if np.mod(counter, 600) == 0:
                    self.save(params.checkpoint_dir, counter)


        except Exception as e:
            print('Done training -- epoch limit reached')
            print(e)
            if counter > 0:
                self.save(params.checkpoint_dir, counter) # save model again
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)
        # END of train()


    def encoder(self, imgs_in, reuse=False):
        """
        returns: 1D vector f1 with size=self.feature_size
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s0 = lrelu(instance_norm(conv2d(imgs_in, self.df_dim, k_h=4, k_w=4, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, self.df_dim * 2, k_h=4, k_w=4, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, self.df_dim * 4, k_h=4, k_w=4, name='g_1_conv2')))
        s3 = lrelu(instance_norm(conv2d(s2, self.df_dim * 8, k_h=2, k_w=2, name='g_1_conv3')))
        s4 = lrelu(instance_norm(conv2d(s3, self.df_dim * 8, k_h=2, k_w=2, name='g_1_conv4')))
        imgs_out = lrelu((linear(tf.reshape(s4, [self.batch_size, -1]), self.feature_size, 'g_1_fc')))

        return imgs_out


    def decoder_image(self, representations, reuse=False):
        """
        returns: batch of images with size 256x60x60x3
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshape = tf.reshape(representations,[self.batch_size, 1, 1, NUM_TILES * self.feature_size])
        # TODO consider increasing capacity of decoder since feature_size-dim is NUM_TILES bigger...
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


    def make_summary_ops(self, g_loss_comp):
        # tf.summary.scalar('g_loss', self.g_loss)
        # tf.summary.scalar('g_loss_comp', g_loss_comp)
        # tf.summary.scalar('cls_loss', self.cls_loss)
        # tf.summary.scalar('dsc_loss_fake', self.dsc_loss_fake)
        # tf.summary.scalar('dsc_loss_real', self.dsc_loss_real)
        tf.summary.scalar('rec_loss_x1hat_x1', self.rec_loss_x1hat_x1)
        # tf.summary.scalar('rec_loss_x4_x1', self.rec_loss_x4_x1)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.model_name)
        get_pp().pprint('Save model to {} with step={}'.format(path, step))
        self.saver.save(self.sess, path, global_step=step)

    def load(self, params, iteration=None):
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(params.log_dir, params.continue_from, params.checkpoint_folder)
        print('Loading variables from ' + checkpoint_dir)

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
        params.continue_from_file = ckpt_file
        print('Reading variables to be restored from ' + ckpt_file)
        self.saver.restore(self.sess, ckpt_file)
        return ckpt_name
