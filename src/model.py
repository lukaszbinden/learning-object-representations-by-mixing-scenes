import os
import signal
from ops_alex import *
from utils_dcgan import *
from utils_common import *
# from input_pipeline_rendered_data import get_pipeline_training_from_dump
from input_pipeline import *
from constants import *
import socket
import numpy as np
tfd = tf.contrib.distributions



class DCGAN(object):

    def __init__(self, sess, params,
                 batch_size=256, sample_size = 64, epochs=1000, image_shape=[256, 256, 3],
                 y_dim=None, z_dim=0, gf_dim=128, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1,
                 is_train=True):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [128]
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
        """ gf_dim: Dimension of gen (ie decoder of AE) filters in first conv layer. [128] """
        self.df_dim = df_dim
        """ df_dim: Dimension of discrim (ie Dsc + encoder of AE) filters in first conv layer. [64] """

        self.gfc_dim = gfc_dim
        """ as of 28.9: not used """
        self.dfc_dim = dfc_dim
        """ as of 28.9: not used """

        self.c_dim = c_dim
        """ c_dim: Dimension of image color. [3] """
        self.cg_dim = cg_dim
        """ as of 28.9: not used """

        self.params = params

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

        self.end = False

        self.build_model()


    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        image_size = self.image_size

        file_train = 'datasets/coco/2017_training/tfrecords/' if 'node0' in socket.gethostname() else 'data/2017_train_small_anys.tfrecords'

        ####################################################################################
        # IMAGE PREPROCESSING ACCORDING TO MEETING 04.10.2018
        reader = tf.TFRecordReader()
        rrm_fn = lambda name : read_record_max(name, reader)
        _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrm_fn)
        self.images_I1 = train_images
        rrs_def_fn = lambda name, scale : read_record_scale(name, reader, scale)
        rrs_fn = lambda name : rrs_def_fn(name, 9) # 90% scale
        _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        self.images_i2 = train_images
        rrs_fn = lambda name : rrs_def_fn(name, 8) # 80% scale
        _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        self.images_i3 = train_images
        rrs_fn = lambda name : rrs_def_fn(name, 7) # 70% scale
        _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        self.images_i4 = train_images
        rrs_fn = lambda name : rrs_def_fn(name, 6) # 60% scale
        _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        self.images_i5 = train_images
        rrs_fn = lambda name : rrs_def_fn(name, 5) # 50% scale
        _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        self.images_i6 = train_images
        rrs_fn = lambda name : rrs_def_fn(name, 4) # 40% scale
        _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        self.images_i7 = train_images

        ####################################################################################
        # self.images_x1 = train_images[0:self.batch_size, :, :, :]
        # """ images_x1: tensor of images (batch_size, image_size, image_size, 3) """
        # self.images_x2 = train_images[self.batch_size:self.batch_size * 2, :, :, :]

        # image overlap arithmetic
        overlap = 0 # self.params.slice_overlap
        # assert overlap, 'hyperparameter \'overlap\' is not an integer'
        slice_size = (image_size + 2 * overlap) / 2
        assert slice_size.is_integer(), 'hyperparameter \'overlap\' invalid: %d' % overlap
        slice_size = int(slice_size)
        slice_size_overlap = slice_size - overlap
        slice_size_overlap = int(slice_size_overlap)
        print('overlap: %d, slice_size: %d, slice_size_overlap: %d' % \
              (overlap, slice_size, slice_size_overlap))

        # create tiles for I1
        self.I1_tile1 = tf.image.crop_to_bounding_box(self.images_I1, 0, 0, slice_size, slice_size)
        self.I1_tile2 = tf.image.crop_to_bounding_box(self.images_I1, 0, slice_size_overlap, slice_size, slice_size)
        self.I1_tile3 = tf.image.crop_to_bounding_box(self.images_I1, slice_size_overlap, 0, slice_size, slice_size)
        self.I1_tile4 = tf.image.crop_to_bounding_box(self.images_I1, slice_size_overlap, slice_size_overlap, slice_size, slice_size)

        # create 1st tile for rest of images
        self.i2_tile1 = tf.image.crop_to_bounding_box(self.images_i2, 0, 0, slice_size, slice_size)
        self.i3_tile1 = tf.image.crop_to_bounding_box(self.images_i3, 0, 0, slice_size, slice_size)
        self.i4_tile1 = tf.image.crop_to_bounding_box(self.images_i4, 0, 0, slice_size, slice_size)
        self.i5_tile1 = tf.image.crop_to_bounding_box(self.images_i5, 0, 0, slice_size, slice_size)
        self.i6_tile1 = tf.image.crop_to_bounding_box(self.images_i6, 0, 0, slice_size, slice_size)
        self.i7_tile1 = tf.image.crop_to_bounding_box(self.images_i7, 0, 0, slice_size, slice_size)

        # create 2nd tile for rest of images
        self.i2_tile2 = tf.image.crop_to_bounding_box(self.images_i2, 0, slice_size_overlap, slice_size, slice_size)
        self.i3_tile2 = tf.image.crop_to_bounding_box(self.images_i3, 0, slice_size_overlap, slice_size, slice_size)
        self.i4_tile2 = tf.image.crop_to_bounding_box(self.images_i4, 0, slice_size_overlap, slice_size, slice_size)
        self.i5_tile2 = tf.image.crop_to_bounding_box(self.images_i5, 0, slice_size_overlap, slice_size, slice_size)
        self.i6_tile2 = tf.image.crop_to_bounding_box(self.images_i6, 0, slice_size_overlap, slice_size, slice_size)
        self.i7_tile2 = tf.image.crop_to_bounding_box(self.images_i7, 0, slice_size_overlap, slice_size, slice_size)

        # create 3rd tile for rest of images
        self.i2_tile3 = tf.image.crop_to_bounding_box(self.images_i2, slice_size_overlap, 0, slice_size, slice_size)
        self.i3_tile3 = tf.image.crop_to_bounding_box(self.images_i3, slice_size_overlap, 0, slice_size, slice_size)
        self.i4_tile3 = tf.image.crop_to_bounding_box(self.images_i4, slice_size_overlap, 0, slice_size, slice_size)
        self.i5_tile3 = tf.image.crop_to_bounding_box(self.images_i5, slice_size_overlap, 0, slice_size, slice_size)
        self.i6_tile3 = tf.image.crop_to_bounding_box(self.images_i6, slice_size_overlap, 0, slice_size, slice_size)
        self.i7_tile3 = tf.image.crop_to_bounding_box(self.images_i7, slice_size_overlap, 0, slice_size, slice_size)

        # create 4th tile for rest of images
        self.i2_tile4 = tf.image.crop_to_bounding_box(self.images_i2, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        self.i3_tile4 = tf.image.crop_to_bounding_box(self.images_i3, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        self.i4_tile4 = tf.image.crop_to_bounding_box(self.images_i4, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        self.i5_tile4 = tf.image.crop_to_bounding_box(self.images_i5, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        self.i6_tile4 = tf.image.crop_to_bounding_box(self.images_i6, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        self.i7_tile4 = tf.image.crop_to_bounding_box(self.images_i7, slice_size_overlap, slice_size_overlap, slice_size, slice_size)

        self.chunk_num = self.params.chunk_num
        """ number of chunks: 8 """
        self.chunk_size = self.params.chunk_size
        """ size per chunk: 64 """
        self.feature_size = self.chunk_size*self.chunk_num
        """ equals the size of all chunks from a single tile """

        with tf.variable_scope('generator') as scope_generator:
            self.I1_f_1 = self.encoder(self.I1_tile1)

            self.f_I1_composite = tf.zeros((self.batch_size, NUM_TILES_L2_MIX * self.feature_size))
            # this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.decoder(self.f_I1_composite)

            # Classifier
            # -> this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.classifier(self.I1_tile1, self.I1_tile1, self.I1_tile1, self.I1_tile1
                            , self.I1_tile1, self.I1_tile1, self.I1_tile1, self.I1_tile1
                            , self.I1_tile1, self.I1_tile1, self.I1_tile1, self.I1_tile1)

            # to share the weights between the Encoders
            scope_generator.reuse_variables()

            self.I1_f_2 = self.encoder(self.I1_tile2)
            self.I1_f_3 = self.encoder(self.I1_tile3)
            self.I1_f_4 = self.encoder(self.I1_tile4)

            self.i2_f_1 = self.encoder(self.i2_tile1)
            self.i3_f_1 = self.encoder(self.i3_tile1)
            self.i4_f_1 = self.encoder(self.i4_tile1)
            self.i5_f_1 = self.encoder(self.i5_tile1)
            self.i6_f_1 = self.encoder(self.i6_tile1)
            self.i7_f_1 = self.encoder(self.i7_tile1)

            self.i2_f_2 = self.encoder(self.i2_tile2)
            self.i3_f_2 = self.encoder(self.i3_tile2)
            self.i4_f_2 = self.encoder(self.i4_tile2)
            self.i5_f_2 = self.encoder(self.i5_tile2)
            self.i6_f_2 = self.encoder(self.i6_tile2)
            self.i7_f_2 = self.encoder(self.i7_tile2)

            self.i2_f_3 = self.encoder(self.i2_tile3)
            self.i3_f_3 = self.encoder(self.i3_tile3)
            self.i4_f_3 = self.encoder(self.i4_tile3)
            self.i5_f_3 = self.encoder(self.i5_tile3)
            self.i6_f_3 = self.encoder(self.i6_tile3)
            self.i7_f_3 = self.encoder(self.i7_tile3)

            self.i2_f_4 = self.encoder(self.i2_tile4)
            self.i3_f_4 = self.encoder(self.i3_tile4)
            self.i4_f_4 = self.encoder(self.i4_tile4)
            self.i5_f_4 = self.encoder(self.i5_tile4)
            self.i6_f_4 = self.encoder(self.i6_tile4)
            self.i7_f_4 = self.encoder(self.i7_tile4)

            # choose tile with miminum L2 distance to I1 tile1
            with tf.variable_scope('L2_tile1_selection'):
                rec_loss_f1_f2 = tf.reduce_mean(tf.square(self.I1_f_1 - self.i2_f_1), 1)
                rec_loss_f1_f3 = tf.reduce_mean(tf.square(self.I1_f_1 - self.i3_f_1), 1)
                rec_loss_f1_f4 = tf.reduce_mean(tf.square(self.I1_f_1 - self.i4_f_1), 1)
                rec_loss_f1_f5 = tf.reduce_mean(tf.square(self.I1_f_1 - self.i5_f_1), 1)
                rec_loss_f1_f6 = tf.reduce_mean(tf.square(self.I1_f_1 - self.i6_f_1), 1)
                rec_loss_f1_f7 = tf.reduce_mean(tf.square(self.I1_f_1 - self.i7_f_1), 1)

                all = tf.stack(axis=0, values=[rec_loss_f1_f2, rec_loss_f1_f3, rec_loss_f1_f4, rec_loss_f1_f5, rec_loss_f1_f6, rec_loss_f1_f7])
                ind = tf.argmin(all, axis=0)
                all_tile1 = tf.concat([self.i2_tile1, self.i3_tile1, self.i4_tile1, self.i5_tile1, self.i6_tile1, self.i7_tile1], axis=0)
                all_f1 = tf.concat([self.i2_f_1, self.i3_f_1, self.i4_f_1, self.i5_f_1, self.i6_f_1, self.i7_f_1], axis=0)
                for i in range(self.batch_size):
                    # choose which batch i holds the L2-closest tile (i.e. w/ lowest L2 loss), then choose that tile at position i
                    tile = all_tile1[ind[i] * self.batch_size:(ind[i] + 1) * self.batch_size][i]
                    tile = tf.expand_dims(tile, 0)
                    self.J_1_tile = tile if i == 0 else tf.concat(axis=0, values=[self.J_1_tile, tile])
                    f = all_f1[ind[i] * self.batch_size:(ind[i] + 1) * self.batch_size][i]
                    f = tf.expand_dims(f, 0)
                    self.J_1_f = f if i == 0 else tf.concat(axis=0, values=[self.J_1_f, f])

                assert self.J_1_tile.shape[0] == self.batch_size
                assert self.J_1_tile.shape[1] == int(self.image_size / 2)
                assert self.J_1_tile.shape[2] == int(self.image_size / 2)
                assert self.J_1_tile.shape[3] == self.c_dim

                # all = tf.stack([rec_loss_f1_f2, rec_loss_f1_f3, rec_loss_f1_f4, rec_loss_f1_f5, rec_loss_f1_f6, rec_loss_f1_f7])
                # ind = tf.argmin(all, axis=0)
                # all_tile1 = tf.concat([self.i2_tile1, self.i3_tile1, self.i4_tile1, self.i5_tile1, self.i6_tile1, self.i7_tile1], axis=0)
                # self.J_1_tile = all_tile1[ind * self.batch_size:(ind + 1) * self.batch_size]
                # tf.ensure_shape(self.J_1_tile, (self.batch_size, None, None, None))
                # all_f1 = tf.concat([self.i2_f_1, self.i3_f_1, self.i4_f_1, self.i5_f_1, self.i6_f_1, self.i7_f_1], axis=0)
                # self.J_1_f = all_f1[ind * self.batch_size:(ind + 1) * self.batch_size]

            # choose tile with miminum L2 distance to I1 tile2
            with tf.variable_scope('L2_tile2_selection'):
                rec_loss_f2_f2 = tf.reduce_mean(tf.square(self.I1_f_2 - self.i2_f_2), 1)
                rec_loss_f2_f3 = tf.reduce_mean(tf.square(self.I1_f_2 - self.i3_f_2), 1)
                rec_loss_f2_f4 = tf.reduce_mean(tf.square(self.I1_f_2 - self.i4_f_2), 1)
                rec_loss_f2_f5 = tf.reduce_mean(tf.square(self.I1_f_2 - self.i5_f_2), 1)
                rec_loss_f2_f6 = tf.reduce_mean(tf.square(self.I1_f_2 - self.i6_f_2), 1)
                rec_loss_f2_f7 = tf.reduce_mean(tf.square(self.I1_f_2 - self.i7_f_2), 1)

                all = tf.stack(axis=0, values=[rec_loss_f2_f2, rec_loss_f2_f3, rec_loss_f2_f4, rec_loss_f2_f5, rec_loss_f2_f6, rec_loss_f2_f7])
                ind = tf.argmin(all, axis=0)
                all_tile2 = tf.concat([self.i2_tile2, self.i3_tile2, self.i4_tile2, self.i5_tile2, self.i6_tile2, self.i7_tile2], axis=0)
                all_f2 = tf.concat([self.i2_f_2, self.i3_f_2, self.i4_f_2, self.i5_f_2, self.i6_f_2, self.i7_f_2], axis=0)
                for i in range(self.batch_size):
                    # choose which batch i holds the L2-closest tile (i.e. w/ lowest L2 loss), then choose that tile at position i
                    tile = all_tile2[ind[i] * self.batch_size:(ind[i] + 1) * self.batch_size][i]
                    tile = tf.expand_dims(tile, 0)
                    self.J_2_tile = tile if i == 0 else tf.concat(axis=0, values=[self.J_2_tile, tile])
                    f = all_f2[ind[i] * self.batch_size:(ind[i] + 1) * self.batch_size][i]
                    f = tf.expand_dims(f, 0)
                    self.J_2_f = f if i == 0 else tf.concat(axis=0, values=[self.J_2_f, f])

                assert self.J_2_tile.shape[0] == self.batch_size
                assert self.J_2_tile.shape[1] == int(self.image_size / 2)
                assert self.J_2_tile.shape[2] == int(self.image_size / 2)
                assert self.J_2_tile.shape[3] == self.c_dim

                # all = tf.stack([rec_loss_f2_f2, rec_loss_f2_f3, rec_loss_f2_f4, rec_loss_f2_f5, rec_loss_f2_f6, rec_loss_f2_f7])
                # ind = tf.argmin(all, axis=0)
                # all_tile2 = tf.concat([self.i2_tile2, self.i3_tile2, self.i4_tile2, self.i5_tile2, self.i6_tile2, self.i7_tile2], axis=0)
                # self.J_2_tile = all_tile2[ind * self.batch_size:(ind + 1) * self.batch_size]
                # tf.ensure_shape(self.J_2_tile, (self.batch_size, None, None, None))
                # all_f2 = tf.concat([self.i2_f_2, self.i3_f_2, self.i4_f_2, self.i5_f_2, self.i6_f_2, self.i7_f_2], axis=0)
                # self.J_2_f = all_f2[ind * self.batch_size:(ind + 1) * self.batch_size]

            # choose tile with miminum L2 distance to I1 tile3
            with tf.variable_scope('L2_tile3_selection'):
                rec_loss_f3_f2 = tf.reduce_mean(tf.square(self.I1_f_3 - self.i2_f_3), 1)
                rec_loss_f3_f3 = tf.reduce_mean(tf.square(self.I1_f_3 - self.i3_f_3), 1)
                rec_loss_f3_f4 = tf.reduce_mean(tf.square(self.I1_f_3 - self.i4_f_3), 1)
                rec_loss_f3_f5 = tf.reduce_mean(tf.square(self.I1_f_3 - self.i5_f_3), 1)
                rec_loss_f3_f6 = tf.reduce_mean(tf.square(self.I1_f_3 - self.i6_f_3), 1)
                rec_loss_f3_f7 = tf.reduce_mean(tf.square(self.I1_f_3 - self.i7_f_3), 1)

                all = tf.stack(axis=0, values=[rec_loss_f3_f2, rec_loss_f3_f3, rec_loss_f3_f4, rec_loss_f3_f5, rec_loss_f3_f6, rec_loss_f3_f7])
                ind = tf.argmin(all, axis=0)
                all_tile3 = tf.concat([self.i2_tile3, self.i3_tile3, self.i4_tile3, self.i5_tile3, self.i6_tile3, self.i7_tile3], axis=0)
                all_f3 = tf.concat([self.i2_f_3, self.i3_f_3, self.i4_f_3, self.i5_f_3, self.i6_f_3, self.i7_f_3], axis=0)
                for i in range(self.batch_size):
                    # choose which batch i holds the L2-closest tile (i.e. w/ lowest L2 loss), then choose that tile at position i
                    tile = all_tile3[ind[i] * self.batch_size:(ind[i] + 1) * self.batch_size][i]
                    tile = tf.expand_dims(tile, 0)
                    self.J_3_tile = tile if i == 0 else tf.concat(axis=0, values=[self.J_3_tile, tile])
                    f = all_f3[ind[i] * self.batch_size:(ind[i] + 1) * self.batch_size][i]
                    f = tf.expand_dims(f, 0)
                    self.J_3_f = f if i == 0 else tf.concat(axis=0, values=[self.J_3_f, f])

                assert self.J_3_tile.shape[0] == self.batch_size
                assert self.J_3_tile.shape[1] == int(self.image_size / 2)
                assert self.J_3_tile.shape[2] == int(self.image_size / 2)
                assert self.J_3_tile.shape[3] == self.c_dim
                # tf.ensure_shape(self.J_3_tile, (self.batch_size, int(self.image_size / 2), int(self.image_size / 2), self.c_dim))
                # tf.ensure_shape(self.J_3_f, (self.batch_size, self.feature_size))

            # choose tile with miminum L2 distance to I1 tile4
            with tf.variable_scope('L2_tile4_selection'):
                rec_loss_f4_f2 = tf.reduce_mean(tf.square(self.I1_f_4 - self.i2_f_4), 1)
                rec_loss_f4_f3 = tf.reduce_mean(tf.square(self.I1_f_4 - self.i3_f_4), 1)
                rec_loss_f4_f4 = tf.reduce_mean(tf.square(self.I1_f_4 - self.i4_f_4), 1)
                rec_loss_f4_f5 = tf.reduce_mean(tf.square(self.I1_f_4 - self.i5_f_4), 1)
                rec_loss_f4_f6 = tf.reduce_mean(tf.square(self.I1_f_4 - self.i6_f_4), 1)
                rec_loss_f4_f7 = tf.reduce_mean(tf.square(self.I1_f_4 - self.i7_f_4), 1)

                all = tf.stack(axis=0, values=[rec_loss_f4_f2, rec_loss_f4_f3, rec_loss_f4_f4, rec_loss_f4_f5, rec_loss_f4_f6, rec_loss_f4_f7])
                ind = tf.argmin(all, axis=0)
                all_tile4 = tf.concat([self.i2_tile4, self.i3_tile4, self.i4_tile4, self.i5_tile4, self.i6_tile4, self.i7_tile4], axis=0)
                all_f4 = tf.concat([self.i2_f_4, self.i3_f_4, self.i4_f_4, self.i5_f_4, self.i6_f_4, self.i7_f_4], axis=0)
                for i in range(self.batch_size):
                    # choose which batch i holds the L2-closest tile (i.e. w/ lowest L2 loss), then choose that tile at position i
                    tile = all_tile4[ind[i] * self.batch_size:(ind[i] + 1) * self.batch_size][i]
                    tile = tf.expand_dims(tile, 0)
                    self.J_4_tile = tile if i == 0 else tf.concat(axis=0, values=[self.J_4_tile, tile])
                    f = all_f4[ind[i] * self.batch_size:(ind[i] + 1) * self.batch_size][i]
                    f = tf.expand_dims(f, 0)
                    self.J_4_f = f if i == 0 else tf.concat(axis=0, values=[self.J_4_f, f])

                assert self.J_4_tile.shape[0] == self.batch_size
                assert self.J_4_tile.shape[1] == int(self.image_size / 2)
                assert self.J_4_tile.shape[2] == int(self.image_size / 2)
                assert self.J_4_tile.shape[3] == self.c_dim
                # tf.ensure_shape(self.J_4_tile, (self.batch_size, int(self.image_size / 2), int(self.image_size / 2), self.c_dim))
                # tf.ensure_shape(self.J_4_f, (self.batch_size, self.feature_size))


            # having all 4 selected tiles J_*, assemble the equivalent of images_I2 analogous to images_I1
            row1 = tf.concat([self.J_1_tile, self.J_3_tile], axis=1)
            row2 = tf.concat([self.J_2_tile, self.J_4_tile], axis=1)
            self.images_I2 = tf.concat([row1, row2], axis=2)
            assert self.images_I2.shape[1] == self.images_I1.shape[1]
            assert self.images_I2.shape[2] == self.images_I1.shape[2]
            assert self.images_I2.shape[3] == self.images_I1.shape[3]

            # build composite feature including all I1 tile features
            self.f_I1_composite = tf.concat([self.I1_f_1, self.I1_f_2, self.I1_f_3, self.I1_f_4], 1)
            self.images_I1_hat = self.decoder(self.f_I1_composite)
            assert self.images_I1_hat.shape[1] == self.image_size
            # Enc/Dec for I1 __end ##########################################

            # Enc/Dec for I2 __start ##########################################
            # build composite feature including all I2 tile features
            self.f_I2_composite = tf.concat([self.J_1_f, self.J_2_f, self.J_3_f, self.J_4_f], 1)
            self.images_I2_hat = self.decoder(self.f_I2_composite)
            assert self.images_I2_hat.shape == self.images_I1.shape
            # Enc/Dec for I2 __end ##########################################

            # Mask handling __start ##########################################
            # for the mask e.g. [0 1 1 0], of shape (4,)
            # 1 selects the corresponding tile from I1
            # 0 selects the corresponding tile from I2
            self.mask = tfd.Bernoulli(self.params.mask_bias_x1).sample(NUM_TILES_L2_MIX)

            # each tile chunk is initialized with 1's (64,256)
            a_tile_chunk = tf.ones((self.batch_size,self.feature_size),dtype=tf.int32)
            assert a_tile_chunk.shape[0] == self.batch_size
            assert a_tile_chunk.shape[1] == self.feature_size

            # mix the tile features according to the mask m
            # for each tile slot in f_1_2 fill it from either x1 or x2
            # tile_feature = includes all chunks from the same tile
            for tile_id in range(0, NUM_TILES_L2_MIX): # for each tile feature slot
                t_f_I1_tile_feature = self.f_I1_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                assert t_f_I1_tile_feature.shape[0] == a_tile_chunk.shape[0]
                assert t_f_I1_tile_feature.shape[1] == self.feature_size
                t_f_I2_tile_feature = self.f_I2_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                assert t_f_I2_tile_feature.shape[1] == self.feature_size
                assert t_f_I2_tile_feature.shape[1] == a_tile_chunk.shape[1]
                tile_mask_batchsize = tf.equal(self.mask[tile_id] * a_tile_chunk, FROM_I1)
                assert tile_mask_batchsize.shape[0] == self.batch_size
                assert tile_mask_batchsize.shape[1] == self.feature_size
                assert tile_mask_batchsize.shape == t_f_I1_tile_feature.shape
                assert tile_mask_batchsize.shape[1] == t_f_I2_tile_feature.shape[1]
                f_feature_selected = tf.where(tile_mask_batchsize, t_f_I1_tile_feature, t_f_I2_tile_feature)
                self.f_I1_I2_mix = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_I1_I2_mix, f_feature_selected])

            assert self.f_I1_I2_mix.shape[0] == self.batch_size
            assert self.f_I1_I2_mix.shape[1] == self.feature_size * NUM_TILES_L2_MIX

            # Dec I1I2
            self.images_I1I2_mix = self.decoder(self.f_I1_I2_mix)

            # create tiles for I1I2
            self.I1I2_tile1 = tf.image.crop_to_bounding_box(self.images_I1I2_mix, 0, 0, slice_size, slice_size)
            self.I1I2_tile2 = tf.image.crop_to_bounding_box(self.images_I1I2_mix, 0, slice_size_overlap, slice_size, slice_size)
            self.I1I2_tile3 = tf.image.crop_to_bounding_box(self.images_I1I2_mix, slice_size_overlap, 0, slice_size, slice_size)
            self.I1I2_tile4 = tf.image.crop_to_bounding_box(self.images_I1I2_mix, slice_size_overlap, slice_size_overlap, slice_size, slice_size)

            # Cls (input tiles_I1, tiles_I2, tiles_I1I2)
            self.mask_predicted = self.classifier(self.I1_tile1, self.I1_tile2, self.I1_tile3, self.I1_tile4
                                                  , self.J_1_tile, self.J_2_tile, self.J_3_tile, self.J_4_tile
                                                  , self.I1I2_tile1, self.I1I2_tile2, self.I1I2_tile3, self.I1I2_tile4)
            """ cls is of size (batch_size, 4) """
            assert self.mask_predicted.shape[0] == self.batch_size
            assert self.mask_predicted.shape[1] == NUM_TILES_L2_MIX

            # cf original mask
            self.mask_actual = tf.cast(tf.ones((self.batch_size, NUM_TILES_L2_MIX), dtype=tf.int32) * self.mask, tf.float32)
            """ mask_actual: mask (4,) scaled to batch_size, of shape (64, 4) """
            assert self.mask_predicted.shape == self.mask_actual.shape

            # f3 (Enc for f3)
            self.I1I2_f_1 = self.encoder(self.I1I2_tile1)
            self.I1I2_f_2 = self.encoder(self.I1I2_tile2)
            self.I1I2_f_3 = self.encoder(self.I1I2_tile3)
            self.I1I2_f_4 = self.encoder(self.I1I2_tile4)
            assert self.I1I2_f_4.shape == self.I1I2_f_1.shape

            # build composite feature including all I1 tile features
            self.f_I1_I2_mix_hat = tf.concat([self.I1I2_f_1, self.I1I2_f_2, self.I1I2_f_3, self.I1I2_f_4], 1)
            assert self.f_I1_I2_mix_hat.shape == self.f_I1_I2_mix.shape
            assert self.f_I1_I2_mix_hat.shape[1] == self.feature_size * NUM_TILES_L2_MIX

            # RECONSTRUCT f_I1_composite_hat/f_I2_composite_hat FROM f_I1_I2_mix_hat START
            for tile_id in range(0, NUM_TILES_L2_MIX):
                f_mix_tile_feature = self.f_I1_I2_mix_hat[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                t_f_I1_tile_feature = self.f_I1_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                t_f_I2_tile_feature = self.f_I2_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                f_feature_selected = tf.where(tf.equal(self.mask[tile_id] * a_tile_chunk, FROM_I1), f_mix_tile_feature, t_f_I1_tile_feature)
                assert f_feature_selected.shape[1] == a_tile_chunk.shape[1]
                self.f_I1_composite_hat = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_I1_composite_hat, f_feature_selected])
                f_feature_selected = tf.where(tf.equal(self.mask[tile_id] * a_tile_chunk, FROM_I2), f_mix_tile_feature, t_f_I2_tile_feature)
                assert f_feature_selected.shape[1] == a_tile_chunk.shape[1]
                self.f_I2_composite_hat = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_I2_composite_hat, f_feature_selected])

            assert self.f_I1_composite_hat.shape[0] == self.batch_size
            assert self.f_I1_composite_hat.shape[1] == self.feature_size * NUM_TILES_L2_MIX
            assert self.f_I1_composite_hat.shape == self.f_I1_composite.shape
            assert self.f_I2_composite_hat.shape[1] == self.f_I2_composite.shape[1]
            # RECONSTRUCT f_I1_composite_hat/f_I2_composite_hat FROM f_I1_I2_mix_hat END

            # decode to I4 for L2 with I1
            self.images_I4 = self.decoder(self.f_I1_composite_hat)
            """ images_I4: batch of reconstructed images I4 with shape (batch_size, 128, 128, 3) """
            # decode to I5 for L2 with I2
            self.images_I5 = self.decoder(self.f_I2_composite_hat)

        with tf.variable_scope('classifier_loss'):
            # Cls loss; mask_batchsize here is GT, cls should predict correct mask..
            self.cls_loss = binary_cross_entropy_with_logits(self.mask_actual, self.mask_predicted)
            """ cls_loss: a scalar, of shape () """

        with tf.variable_scope('discriminator'):
            # Dsc for I1
            self.dsc_I1 = self.discriminator(self.images_I1)
            """ Dsc_I1: real/fake, of shape (64, 1) """
            # Dsc for I3
            self.dsc_I1I2 = self.discriminator(self.images_I1I2_mix, reuse=True)
            """ Dsc_I1I2: real/fake, of shape (64, 1) """

        with tf.variable_scope('discriminator_loss'):
            # Dsc loss x1
            self.dsc_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_I1), self.dsc_I1)
            # Dsc loss x3
            # this is max_D part of minmax loss function
            self.dsc_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.dsc_I1I2), self.dsc_I1I2)
            self.dsc_loss = self.dsc_loss_real + self.dsc_loss_fake
            """ dsc_loss: a scalar, of shape () """

        with tf.variable_scope('generator_loss'):
            # D (fix Dsc you have loss for G) -> cf. Dec
            # images_x3 = Dec(f_1_2) = G(f_1_2); Dsc(images_x3) = dsc_x3
            # rationale behind g_loss: this is min_G part of minmax loss function: min log D(G(x))
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_I1I2), self.dsc_I1I2)

        with tf.variable_scope('L2') as _:
            # Reconstruction loss L2 between I1 and I1' (to ensure autoencoder works properly)
            self.rec_loss_I1hat_I1 = tf.reduce_mean(tf.square(self.images_I1_hat - self.images_I1))
            """ rec_loss_x1hat_x1: a scalar, of shape () """
            # Reconstruction loss L2 between I2 and I2' (to ensure autoencoder works properly)
            self.rec_loss_I2hat_I2 = tf.reduce_mean(tf.square(self.images_I2_hat - self.images_I2))
            # L2 between I1 and I4
            self.rec_loss_I4_I1 = tf.reduce_mean(tf.square(self.images_I4 - self.images_I1))
            # L2 between I2 and I5
            self.rec_loss_I5_I2 = tf.reduce_mean(tf.square(self.images_I5 - self.images_I2))

        # TODO what for?
        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()
        # Tf stuff (tell variables how to train..)
        self.dsc_vars = [var for var in t_vars if 'd_' in var.name] # discriminator
        self.gen_vars = [var for var in t_vars if 'g_' in var.name] # encoder + decoder (generator)
        self.cls_vars = [var for var in t_vars if 'c_' in var.name] # classifier

        # save the weights
        self.saver = tf.train.Saver(self.dsc_vars + self.gen_vars + self.cls_vars + batch_norm.shadow_variables, max_to_keep=5)
        # END of build_model

    def train(self, params):
        """Train DCGAN"""

        if params.continue_from_iteration:
            iteration = params.continue_from_iteration
        else:
            iteration = 0

        global_step = tf.Variable(iteration, name='global_step', trainable=False)

        if params.learning_rate_generator:
            self.g_learning_rate = params.learning_rate_generator
        else:
            self.g_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        if params.learning_rate_discriminator:
            self.d_learning_rate = params.learning_rate_discriminator
        else:
            self.d_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        self.c_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        print('g_learning_rate: %s' % self.g_learning_rate)
        print('d_learning_rate: %s' % self.d_learning_rate)

        g_loss_comp = 5 * self.rec_loss_I1hat_I1 + 5 * self.rec_loss_I2hat_I2 + 5 * self.rec_loss_I4_I1 + 5 * self.rec_loss_I5_I2 + 1 * self.g_loss + 1 * self.cls_loss
        # for autoencoder
        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=params.beta1, beta2=params.beta2) \
                          .minimize(g_loss_comp, var_list=self.gen_vars) # includes encoder + decoder weights
        # for classifier
        c_optim = tf.train.AdamOptimizer(learning_rate=self.c_learning_rate, beta1=0.5) \
                          .minimize(self.cls_loss, var_list=self.cls_vars)  # params.beta1
        # for Dsc
        d_optim = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=params.beta1, beta2=params.beta2) \
                          .minimize(self.dsc_loss, var_list=self.dsc_vars, global_step=global_step)

        # what you specify in the argument to control_dependencies is ensured to be evaluated before anything you define in the with block
        with tf.control_dependencies([g_optim]):
            # this is also part of BP/training; this line is a fix re BN acc. to Stackoverflow
            g_optim = tf.group(self.bn_assigners)

        tf.global_variables_initializer().run()
        if params.continue_from:
            ckpt_name = self.load(params, params.continue_from_iteration)
            iteration = int(ckpt_name[ckpt_name.rfind('-')+1:])
            print('continuing from \'%s\'...' % ckpt_name)
            global_step.load(iteration) # load new initial value into variable

        # simple mechanism to coordinate the termination of a set of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        self.make_summary_ops(g_loss_comp)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(params.summary_dir)
        summary_writer.add_graph(self.sess.graph)

        update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS)

        try:
            signal.signal(signal.SIGTERM, self.handle_exit)

            iter_per_epoch = (self.params.num_images / self.batch_size)

            # Training
            while not coord.should_stop():
                # Update D and G network
                self.sess.run([g_optim])
                self.sess.run([c_optim])
                self.sess.run([d_optim])
                iteration += 1
                epoch = iteration / iter_per_epoch
                print('iteration: %s, epoch: %d' % (str(iteration), round(epoch, 2)))

                if iteration % 100 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, iteration)

                if np.mod(iteration, 2000) == 1:
                    self.dump_images(iteration)

                if iteration > 1 and np.mod(iteration, 500) == 0:
                    self.save(params.checkpoint_dir, iteration)

                # for spectral normalization
                for update_op in update_ops:
                    self.sess.run(update_op)

                if self.end:
                    print('going to shutdown now...')
                    self.params.iterations = iteration
                    self.save(params.checkpoint_dir, iteration) # save model again
                    break

        except Exception as e:
            if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Exception here, ending training..')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(e)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iteration > 0:
                self.save(params.checkpoint_dir, iteration) # save model again
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)
        # END of train()

    def path(self, filename):
        return os.path.join(self.params.summary_dir, filename)

    def discriminator(self, image, keep_prob=0.5, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # cf. DCGAN impl https://github.com/carpedm20/DCGAN-tensorflow.git
        h0 = lrelu(conv2d(image, self.df_dim, use_spectral_norm=True, name='d_1_h0_conv'))
        h1 = lrelu(conv2d(h0, self.df_dim*2, use_spectral_norm=True, name='d_1_h1_conv'))

        #################################
        ch = self.df_dim*2
        x = h1
        h1 = attention(x, ch, sn=True, scope="d_attention", reuse=reuse)
        #################################

        h2 = lrelu(conv2d(h1, self.df_dim*4, use_spectral_norm=True, name='d_1_h2_conv'))
        # NB: k=1,d=1 is like an FC layer -> to strengthen h3, to give it more capacity
        h3 = lrelu(conv2d(h2, self.df_dim*8,k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='d_1_h3_conv'))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_1_h3_lin')

        return tf.nn.sigmoid(h4)


    def classifier(self, x1_tile1, x1_tile2, x1_tile3, x1_tile4,
                   x2_tile1, x2_tile2, x2_tile3, x2_tile4,
                   x3_tile1, x3_tile2, x3_tile3, x3_tile4,
                   reuse=False):
        """From paper:
        For the classifier, we use AlexNet with batch normalization after each
        convolutional layer, but we do not use any dropout. The image inputs of
        the classifier are concatenated along the RGB channels.

        returns: a 1D matrix of size NUM_TILES i.e. (batch_size, 9)
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        concatenated = tf.concat(axis=3, values=[x1_tile1, x1_tile2, x1_tile3, x1_tile4])
        concatenated = tf.concat(axis=3, values=[concatenated, x2_tile1, x2_tile2, x2_tile3, x2_tile4])
        concatenated = tf.concat(axis=3, values=[concatenated, x3_tile1, x3_tile2, x3_tile3, x3_tile4])

        conv1 = self.c_bn1(conv(concatenated, 96, 8,8,2,2, padding='VALID', name='c_3_s0_conv'))
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='c_3_mp0')

        conv2 = self.c_bn2(conv(pool1, 256, 5,5,1,1, groups=2, name='c_3_conv2'))
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='c_3_pool2')

        conv3 = self.c_bn3(conv(pool2, 384, 3, 3, 1, 1, name='c_3_conv3'))

        conv4 = self.c_bn4(conv(conv3, 384, 3, 3, 1, 1, groups=2, name='c_3_conv4'))

        conv5 = self.c_bn5(conv(conv4, 256, 3, 3, 1, 1, groups=2, name='c_3_conv5'))
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='c_3_pool5')

        fc6 = tf.nn.relu(linear(tf.reshape(pool5, [self.batch_size, -1]), 4096, 'c_3_fc6') )

        fc7 = tf.nn.relu(linear(tf.reshape(fc6, [self.batch_size, -1]), 4096, 'c_3_fc7') )

        self.fc8 = linear(tf.reshape(fc7, [self.batch_size, -1]), NUM_TILES_L2_MIX, 'c_3_fc8')

        return tf.nn.sigmoid(self.fc8)


    def encoder(self, tile_image, reuse=False):
        """
        returns: 1D vector f1 with size=self.feature_size
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s0 = lrelu(instance_norm(conv2d(tile_image, self.df_dim, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, self.df_dim * 2, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, self.df_dim * 4, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv2')))
        # s3 = lrelu(instance_norm(conv2d(s2, self.df_dim * 4, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv3')))
        s4 = lrelu(instance_norm(conv2d(s2, self.df_dim * 8, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv4')))
        s5 = lrelu(instance_norm(conv2d(s4, self.df_dim * 16, k_h=1, k_w=1, use_spectral_norm=True, name='g_1_conv5')))
        rep = lrelu((linear(tf.reshape(s5, [self.batch_size, -1]), self.feature_size, 'g_1_fc')))

        return rep


    def decoder(self, representations, reuse=False):
        """
        returns: batch of images with size 256x60x60x3
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshape = tf.reshape(representations,[self.batch_size, 1, 1, NUM_TILES_L2_MIX * self.feature_size])

        h = deconv2d(reshape, [self.batch_size, 4, 4, self.gf_dim*4], k_h=4, k_w=4, d_h=1, d_w=1, padding='VALID', use_spectral_norm=True, name='g_de_h')
        h = tf.nn.relu(h)

        h1 = deconv2d(h, [self.batch_size, 8, 8, self.gf_dim*4], use_spectral_norm=True, name='g_h1')
        h1 = tf.nn.relu(instance_norm(h1))

        h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim*4], use_spectral_norm=True, name='g_h2')
        h2 = tf.nn.relu(instance_norm(h2))

        h3 = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim*4], use_spectral_norm=True, name='g_h3')
        h3 = tf.nn.relu(instance_norm(h3))

        #################################
        ch = self.gf_dim*4
        x = h3
        h3 = attention(x, ch, sn=True, scope="g_attention", reuse=reuse)
        #################################

        h4 = deconv2d(h3, [self.batch_size, 64, 64, self.gf_dim*2], use_spectral_norm=True, name='g_h4')
        h4 = tf.nn.relu(instance_norm(h4))

        h5 = deconv2d(h4, [self.batch_size, 128, 128, self.gf_dim*1], use_spectral_norm=True, name='g_h5')
        h5 = tf.nn.relu(instance_norm(h5))

        # h6 = deconv2d(h5, [self.batch_size, 128, 128, self.c_dim], use_spectral_norm=True, name='g_h6')
        # h6 = tf.nn.relu(instance_norm(h6))

        # From https://distill.pub/2016/deconv-checkerboard/
        # - last layer uses stride=1
        # - kernel should be divided by stride to mitigate artifacts
        h6 = deconv2d(h5, [self.batch_size, 128, 128, self.c_dim], k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='g_h7')

        return tf.nn.tanh(h6)


    def make_summary_ops(self, g_loss_comp):
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('g_loss_comp', g_loss_comp)
        tf.summary.scalar('cls_loss', self.cls_loss)
        tf.summary.scalar('dsc_loss', self.dsc_loss)
        tf.summary.scalar('dsc_loss_fake', self.dsc_loss_fake)
        tf.summary.scalar('dsc_loss_real', self.dsc_loss_real)
        tf.summary.scalar('rec_loss_I1hat_I1', self.rec_loss_I1hat_I1)
        tf.summary.scalar('rec_loss_I2hat_I2', self.rec_loss_I2hat_I2)
        tf.summary.scalar('rec_loss_I4_I1', self.rec_loss_I4_I1)
        tf.summary.scalar('rec_loss_I5_I2', self.rec_loss_I5_I2)


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

    def handle_exit(self, signum, frame):
        self.end = True

    def dump_images(self, counter):
        # print out images every so often
        images_x1, images_x2, images_x3, \
        iamges_x4, images_x5, \
        test_mask = \
            self.sess.run([self.images_I1, self.images_I2, self.images_I1I2_mix, \
                           self.images_I4, self.images_I5, \
                           self.mask])
        grid_size = np.ceil(np.sqrt(self.batch_size))
        grid = [grid_size, grid_size]
        save_images(images_x1, grid, self.path('%s_images_I1.jpg' % counter))
        save_images(images_x2, grid, self.path('%s_images_I2.jpg' % counter))
        file_path = self.path('%s_images_I1I2_mix_%s.jpg' % (counter, ''.join(str(e) for e in test_mask)))
        save_images(images_x3, grid, file_path)
        save_images(iamges_x4, grid, self.path('%s_images_I4.jpg' % counter))
        save_images(images_x5, grid, self.path('%s_images_I5.jpg' % counter))
