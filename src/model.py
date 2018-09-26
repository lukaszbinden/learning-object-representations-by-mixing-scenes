import os

from ops_alex import *
from utils_dcgan import *
from utils_common import *
from input_pipeline_rendered_data import get_pipeline_training_from_dump
from scipy.stats import bernoulli
from constants import *
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

        self.build_model()


    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        self.abstract_size = self.sample_size // 2 ** 4
        image_size = self.image_size

        _, _, train_images = get_pipeline_training_from_dump(dump_file='2017_val_small.tfrecords', #'datasets/coco/2017_training/tfrecords/',
                                                                 batch_size=self.batch_size * 2, # for x1 and x2
                                                                 epochs=self.epochs,
                                                                 image_size=image_size,
                                                                 resize_size=image_size,
                                                                 img_channels=self.c_dim)

        _, _, test_images = get_pipeline_training_from_dump(dump_file='2017_val_small.tfrecords', # 'datasets/coco/2017_val/tfrecords/',
                                                                 batch_size=self.batch_size * 2,
                                                                 epochs=10000000, # TODO really?
                                                                 image_size=image_size,
                                                                 resize_size=image_size,
                                                                 img_channels=self.c_dim)

        self.images_x1 = train_images[0:self.batch_size, :, :, :]
        """ images_x1: tensor of images (64, 60, 60, 3) """
        self.images_x2 = train_images[self.batch_size:self.batch_size * 2, :, :, :]

        self.test_images_x1 = test_images[0:self.batch_size, :, :, :]
        self.test_images_x2 = test_images[self.batch_size:self.batch_size * 2, :, :, :]

        # image overlap arithmetic
        overlap = self.params.slice_overlap
        # assert overlap, 'hyperparameter \'overlap\' is not an integer'
        slice_size = (image_size + 2 * overlap) / 3
        assert slice_size.is_integer(), 'hyperparameter \'overlap\' invalid: %d' % overlap
        slice_size = int(slice_size)
        slice_size_overlap = slice_size - overlap
        slice_size_overlap = int(slice_size_overlap)
        print('overlap: %d, slice_size: %d, slice_size_overlap: %d' % \
              (overlap, slice_size, slice_size_overlap))

        # create tiles for x1
        self.x1_tile1_r1c1 = tf.image.crop_to_bounding_box(self.images_x1, 0, 0, slice_size, slice_size)
        self.x1_tile2_r1c2 = tf.image.crop_to_bounding_box(self.images_x1, 0, slice_size_overlap, slice_size, slice_size)
        self.x1_tile3_r1c3 = tf.image.crop_to_bounding_box(self.images_x1, 0, image_size - slice_size, slice_size, slice_size)
        self.x1_tile4_r2c1 = tf.image.crop_to_bounding_box(self.images_x1, slice_size_overlap, 0, slice_size, slice_size)
        self.x1_tile5_r2c2 = tf.image.crop_to_bounding_box(self.images_x1, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        self.x1_tile6_r2c3 = tf.image.crop_to_bounding_box(self.images_x1, slice_size_overlap, image_size - slice_size, slice_size, slice_size)
        self.x1_tile7_r3c1 = tf.image.crop_to_bounding_box(self.images_x1, image_size - slice_size, 0, slice_size, slice_size)
        self.x1_tile8_r3c2 = tf.image.crop_to_bounding_box(self.images_x1, image_size - slice_size, slice_size_overlap, slice_size, slice_size)
        self.x1_tile9_r3c3 = tf.image.crop_to_bounding_box(self.images_x1, image_size - slice_size, image_size - slice_size, slice_size, slice_size)

        # create tiles for x1
        self.x2_tile10_r1c1 = tf.image.crop_to_bounding_box(self.images_x2, 0, 0, slice_size, slice_size)
        self.x2_tile11_r1c2 = tf.image.crop_to_bounding_box(self.images_x2, 0, slice_size_overlap, slice_size, slice_size)
        self.x2_tile12_r1c3 = tf.image.crop_to_bounding_box(self.images_x2, 0, image_size - slice_size, slice_size, slice_size)
        self.x2_tile13_r2c1 = tf.image.crop_to_bounding_box(self.images_x2, slice_size_overlap, 0, slice_size, slice_size)
        self.x2_tile14_r2c2 = tf.image.crop_to_bounding_box(self.images_x2, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        self.x2_tile15_r2c3 = tf.image.crop_to_bounding_box(self.images_x2, slice_size_overlap, image_size - slice_size, slice_size, slice_size)
        self.x2_tile16_r3c1 = tf.image.crop_to_bounding_box(self.images_x2, image_size - slice_size, 0, slice_size, slice_size)
        self.x2_tile17_r3c2 = tf.image.crop_to_bounding_box(self.images_x2, image_size - slice_size, slice_size_overlap, slice_size, slice_size)
        self.x2_tile18_r3c3 = tf.image.crop_to_bounding_box(self.images_x2, image_size - slice_size, image_size - slice_size, slice_size, slice_size)

        self.chunk_num = self.params.chunk_num
        """ number of chunks: 8 """
        self.chunk_size = self.params.chunk_size
        """ size per chunk: 64 """
        self.feature_size = self.chunk_size*self.chunk_num
        """ equals the size of all chunks from a single tile """

        with tf.variable_scope('generator') as scope_generator:
            # Enc/Dec for x1 __start ##########################################
            self.f_1 = self.encoder(self.x1_tile1_r1c1)

            self.f_x1_composite = tf.zeros((self.batch_size, NUM_TILES * self.feature_size))
            # this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.decoder(self.f_x1_composite)

            # Classifier
            # -> this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.classifier(self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1
                            , self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1
                            , self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1, self.x1_tile1_r1c1)

            # to share the weights between the Encoders
            scope_generator.reuse_variables()
            self.f_2 = self.encoder(self.x1_tile2_r1c2)
            self.f_3 = self.encoder(self.x1_tile3_r1c3)
            self.f_4 = self.encoder(self.x1_tile4_r2c1)
            self.f_5 = self.encoder(self.x1_tile5_r2c2)
            self.f_6 = self.encoder(self.x1_tile6_r2c3)
            self.f_7 = self.encoder(self.x1_tile7_r3c1)
            self.f_8 = self.encoder(self.x1_tile8_r3c2)
            self.f_9 = self.encoder(self.x1_tile9_r3c3)

            # build composite feature including all x1 tile features
            self.f_x1_composite = tf.concat([self.f_1, self.f_2, self.f_3, self.f_4, self.f_5, self.f_6, self.f_7, self.f_8, self.f_9], 1)
            # (64, 2304)
            # Dec for x1 -> x1_hat
            self.images_x1_hat = self.decoder(self.f_x1_composite)
            # (64, 300, 300, 3)
            # Enc/Dec for x1 __end ##########################################

            # Enc/Dec for x2 __start ##########################################
            self.f_10 = self.encoder(self.x2_tile10_r1c1)
            self.f_11 = self.encoder(self.x2_tile11_r1c2)
            self.f_12 = self.encoder(self.x2_tile12_r1c3)
            self.f_13 = self.encoder(self.x2_tile13_r2c1)
            self.f_14 = self.encoder(self.x2_tile14_r2c2)
            self.f_15 = self.encoder(self.x2_tile15_r2c3)
            self.f_16 = self.encoder(self.x2_tile16_r3c1)
            self.f_17 = self.encoder(self.x2_tile17_r3c2)
            self.f_18 = self.encoder(self.x2_tile18_r3c3)

            # build composite feature including all x2 tile features
            self.f_x2_composite = tf.concat([self.f_10, self.f_11, self.f_12, self.f_13, self.f_14, self.f_15, self.f_16, self.f_17, self.f_18], 1)
            # Dec for x2 -> x2_hat
            self.images_x2_hat = self.decoder(self.f_x2_composite)
            # Enc/Dec for x2 __end ##########################################

            # Mask handling __start ##########################################
            # for the mask e.g. [0 1 1 0 0 1 1 0 0], of shape (9,)
            # 1 selects the corresponding tile from x1
            # 0 selects the corresponding tile from x2
            # self.mask = bernoulli.rvs(self.params.mask_bias_x1, size=NUM_TILES)
            self.mask = tfd.Bernoulli(self.params.mask_bias_x1).sample(NUM_TILES)
            # self.mask = tf.random_uniform(shape=[NUM_TILES],minval=0,maxval=2,dtype=tf.int32)
            #print('mask: %s' % mask)

            # each tile chunk is initialized with 1's (64,256)
            a_tile_chunk = tf.ones((self.batch_size,self.feature_size),dtype=tf.int32)
            assert a_tile_chunk.shape[0] == self.batch_size
            assert a_tile_chunk.shape[1] == self.feature_size

            ####################### mix each chunk from two image features
            # # chunk stuff: i -> chunk-id
            # i=0
            # f_1_chunk = self.f_1[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            # f_2_chunk = self.f_2[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            #
            # self.f_1_2 = tf.where(tf.equal(mask[i] * a_chunk, 0), f_1_chunk, f_2_chunk)
            #
            # # mix the feature (cf step 2)
            # for i in range(1, self.chunk_num): # for each chunk
            #     f_1_chunk = self.f_1[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            #     f_2_chunk = self.f_2[:, i * self.chunk_size:(i + 1) * self.chunk_size]
            #     self.f_chunk_selected = tf.where(tf.equal(mask[i] * a_chunk, 0), f_1_chunk, f_2_chunk)
            #     self.f_1_2 = tf.concat(axis=1, values=[self.f_1_2, self.f_chunk_selected])
            #######################


            # mix the tile features according to the mask m
            # for each tile slot in f_1_2 fill it from either x1 or x2
            # tile_feature = includes all chunks from the same tile
            for tile_id in range(0, NUM_TILES): # for each tile feature slot
                t_f_x1_tile_feature = self.f_x1_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                assert t_f_x1_tile_feature.shape[1] == self.feature_size
                t_f_x2_tile_feature = self.f_x2_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                assert t_f_x2_tile_feature.shape[1] == self.feature_size
                assert t_f_x2_tile_feature.shape[0] == a_tile_chunk.shape[0]
                assert t_f_x2_tile_feature.shape[1] == a_tile_chunk.shape[1]
                tile_mask_batchsize = tf.equal(self.mask[tile_id] * a_tile_chunk, FROM_X1)
                assert tile_mask_batchsize.shape[0] == self.batch_size
                assert tile_mask_batchsize.shape[1] == self.feature_size
                assert tile_mask_batchsize.shape == t_f_x1_tile_feature.shape
                assert tile_mask_batchsize.shape == t_f_x2_tile_feature.shape
                f_feature_selected = tf.where(tile_mask_batchsize, t_f_x1_tile_feature, t_f_x2_tile_feature)
                self.f_x1_x2_mix = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_x1_x2_mix, f_feature_selected])

            # TODO: put asserts to verify f_1_2 exactly equals corresponding tiles
            # assert tf.where(self.f_1_2[:, 0:])
            assert self.f_x1_x2_mix.shape[0] == self.batch_size
            assert self.f_x1_x2_mix.shape[1] == self.feature_size * NUM_TILES

            # Dec x3
            self.images_x3 = self.decoder(self.f_x1_x2_mix)

            # create tiles for x3
            self.x3_tile1_r1c1 = tf.image.crop_to_bounding_box(self.images_x3, 0, 0, slice_size, slice_size)
            self.x3_tile2_r1c2 = tf.image.crop_to_bounding_box(self.images_x3, 0, slice_size_overlap, slice_size, slice_size)
            self.x3_tile3_r1c3 = tf.image.crop_to_bounding_box(self.images_x3, 0, image_size - slice_size, slice_size, slice_size)
            self.x3_tile4_r2c1 = tf.image.crop_to_bounding_box(self.images_x3, slice_size_overlap, 0, slice_size, slice_size)
            self.x3_tile5_r2c2 = tf.image.crop_to_bounding_box(self.images_x3, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
            self.x3_tile6_r2c3 = tf.image.crop_to_bounding_box(self.images_x3, slice_size_overlap, image_size - slice_size, slice_size, slice_size)
            self.x3_tile7_r3c1 = tf.image.crop_to_bounding_box(self.images_x3, image_size - slice_size, 0, slice_size, slice_size)
            self.x3_tile8_r3c2 = tf.image.crop_to_bounding_box(self.images_x3, image_size - slice_size, slice_size_overlap, slice_size, slice_size)
            self.x3_tile9_r3c3 = tf.image.crop_to_bounding_box(self.images_x3, image_size - slice_size, image_size - slice_size, slice_size, slice_size)

            # Cls (input tiles_x1, tiles_x2, tiles_x3)
            self.mask_predicted = self.classifier(self.x1_tile1_r1c1, self.x1_tile2_r1c2, self.x1_tile3_r1c3, self.x1_tile4_r2c1, self.x1_tile5_r2c2, self.x1_tile6_r2c3, self.x1_tile7_r3c1, self.x1_tile8_r3c2, self.x1_tile9_r3c3
                            , self.x2_tile10_r1c1, self.x2_tile11_r1c2, self.x2_tile12_r1c3, self.x2_tile13_r2c1, self.x2_tile14_r2c2, self.x2_tile15_r2c3, self.x2_tile16_r3c1, self.x2_tile17_r3c2, self.x2_tile18_r3c3
                            , self.x3_tile1_r1c1, self.x3_tile2_r1c2, self.x3_tile3_r1c3, self.x3_tile4_r2c1, self.x3_tile5_r2c2, self.x3_tile6_r2c3, self.x3_tile7_r3c1, self.x3_tile8_r3c2, self.x3_tile9_r3c3)
            """ cls is of size (64, 9) """
            assert self.mask_predicted.shape[0] == self.batch_size
            assert self.mask_predicted.shape[1] == NUM_TILES

            # cf original mask
            self.mask_actual = tf.cast(tf.ones((self.batch_size, NUM_TILES), dtype=tf.int32) * self.mask, tf.float32)
            """ mask_actual: mask (9,) scaled to batch_size, of shape (64, 9) """
            assert self.mask_predicted.shape == self.mask_actual.shape

            # f3 (Enc for f3)
            self.f_3_1 = self.encoder(self.x3_tile1_r1c1)
            self.f_3_2 = self.encoder(self.x3_tile2_r1c2)
            self.f_3_3 = self.encoder(self.x3_tile3_r1c3)
            self.f_3_4 = self.encoder(self.x3_tile4_r2c1)
            self.f_3_5 = self.encoder(self.x3_tile5_r2c2)
            self.f_3_6 = self.encoder(self.x3_tile6_r2c3)
            self.f_3_7 = self.encoder(self.x3_tile7_r3c1)
            self.f_3_8 = self.encoder(self.x3_tile8_r3c2)
            self.f_3_9 = self.encoder(self.x3_tile9_r3c3)
            """ f_3_x: feature rep as 2D vector (batch_size, feature_size) -> (64, 256) """
            assert self.f_3_9.shape == self.f_3_1.shape

            # build composite feature including all x1 tile features
            self.f_x1_x2_mix_hat = tf.concat([self.f_3_1, self.f_3_2, self.f_3_3, self.f_3_4, self.f_3_5, self.f_3_6, self.f_3_7, self.f_3_8, self.f_3_9], 1)
            assert self.f_x1_x2_mix_hat.shape == self.f_x1_x2_mix.shape
            assert self.f_x1_x2_mix_hat.shape[1] == self.feature_size * NUM_TILES

            # # from f3 to f31/f32 START
            # tile_id = 0
            # f_3_chunk = self.f_3[:, tile_id * self.chunk_size:(tile_id + 1) * self.chunk_size]
            # f_1_chunk = self.f_1[:, tile_id * self.chunk_size:(tile_id + 1) * self.chunk_size]
            # f_2_chunk = self.f_2[:, tile_id * self.chunk_size:(tile_id + 1) * self.chunk_size]
            # self.f_3_1 = tf.where(tf.equal(mask[tile_id] * a_chunk, 0), f_3_chunk, f_1_chunk)
            # """ f_3_1: used to be rep_re; of shape (64, 512) """
            # self.f_3_2 = tf.where(tf.equal(mask[tile_id] * a_chunk, 1), f_3_chunk, f_2_chunk)
            # """ f_3_2: used to be repR_re """
            #
            # for tile_id in range(1, self.chunk_num):
            #     f_3_chunk = self.f_3[:, tile_id * self.chunk_size:(tile_id + 1) * self.chunk_size]
            #     f_1_chunk = self.f_1[:, tile_id * self.chunk_size:(tile_id + 1) * self.chunk_size]
            #     f_2_chunk = self.f_2[:, tile_id * self.chunk_size:(tile_id + 1) * self.chunk_size]
            #     self.f_chunk_selected = tf.where(tf.equal(mask[tile_id] * a_chunk, 0), f_3_chunk, f_1_chunk)
            #     self.f_3_1 = tf.concat(axis=1, values=[self.f_3_1, self.f_chunk_selected])
            #     self.f_chunk_selected = tf.where(tf.equal(mask[tile_id] * a_chunk, 1), f_3_chunk, f_2_chunk)
            #     self.f_3_2 = tf.concat(axis=1, values=[self.f_3_2, self.f_chunk_selected])
            # # from f3 to f31/f32 END

            # RECONSTRUCT f_x1_composite_hat/f_x2_composite_hat FROM f_x1_x2_mix_hat START
            for tile_id in range(0, NUM_TILES):
                f_mix_tile_feature = self.f_x1_x2_mix_hat[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                t_f_x1_tile_feature = self.f_x1_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                t_f_x2_tile_feature = self.f_x2_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                f_feature_selected = tf.where(tf.equal(self.mask[tile_id] * a_tile_chunk, FROM_X1), f_mix_tile_feature, t_f_x1_tile_feature)
                assert f_feature_selected.shape[1] == a_tile_chunk.shape[1]
                self.f_x1_composite_hat = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_x1_composite_hat, f_feature_selected])
                """ f_x1_composite_hat: used to be rep_re; of shape (64, 256) """
                f_feature_selected = tf.where(tf.equal(self.mask[tile_id] * a_tile_chunk, FROM_X2), f_mix_tile_feature, t_f_x2_tile_feature)
                assert f_feature_selected.shape[1] == a_tile_chunk.shape[1]
                self.f_x2_composite_hat = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_x2_composite_hat, f_feature_selected])
                """ f_x2_composite_hat: used to be repR_re """

            assert self.f_x1_composite_hat.shape[0] == self.batch_size
            assert self.f_x1_composite_hat.shape[1] == self.feature_size * NUM_TILES
            assert self.f_x1_composite_hat.shape == self.f_x1_composite.shape
            assert self.f_x2_composite_hat.shape == self.f_x2_composite.shape
            # RECONSTRUCT f_x1_composite_hat/f_x2_composite_hat FROM f_x1_x2_mix_hat END

            # decode to x4 for L2 with x1
            self.images_x4 = self.decoder(self.f_x1_composite_hat)
            """ images_x4: batch of reconstructed images x4 with shape (64, 300, 300, 3) """
            # decode to x5 for L2 with x2
            self.images_x5 = self.decoder(self.f_x2_composite_hat)

            ##########################################################################
            ##########################################################################
            # for test only
            ##########################################################################
            # create tiles for test_images_x1
            self.t_x1_tile1_r1c1 = tf.image.crop_to_bounding_box(self.test_images_x1, 0, 0, slice_size, slice_size)
            self.t_x1_tile2_r1c2 = tf.image.crop_to_bounding_box(self.test_images_x1, 0, slice_size_overlap, slice_size, slice_size)
            self.t_x1_tile3_r1c3 = tf.image.crop_to_bounding_box(self.test_images_x1, 0, image_size - slice_size, slice_size, slice_size)
            self.t_x1_tile4_r2c1 = tf.image.crop_to_bounding_box(self.test_images_x1, slice_size_overlap, 0, slice_size, slice_size)
            self.t_x1_tile5_r2c2 = tf.image.crop_to_bounding_box(self.test_images_x1, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
            self.t_x1_tile6_r2c3 = tf.image.crop_to_bounding_box(self.test_images_x1, slice_size_overlap, image_size - slice_size, slice_size, slice_size)
            self.t_x1_tile7_r3c1 = tf.image.crop_to_bounding_box(self.test_images_x1, image_size - slice_size, 0, slice_size, slice_size)
            self.t_x1_tile8_r3c2 = tf.image.crop_to_bounding_box(self.test_images_x1, image_size - slice_size, slice_size_overlap, slice_size, slice_size)
            self.t_x1_tile9_r3c3 = tf.image.crop_to_bounding_box(self.test_images_x1, image_size - slice_size, image_size - slice_size, slice_size, slice_size)
            self.t_f_1 = self.encoder(self.t_x1_tile1_r1c1)
            self.t_f_2 = self.encoder(self.t_x1_tile2_r1c2)
            self.t_f_3 = self.encoder(self.t_x1_tile3_r1c3)
            self.t_f_4 = self.encoder(self.t_x1_tile4_r2c1)
            self.t_f_5 = self.encoder(self.t_x1_tile5_r2c2)
            self.t_f_6 = self.encoder(self.t_x1_tile6_r2c3)
            self.t_f_7 = self.encoder(self.t_x1_tile7_r3c1)
            self.t_f_8 = self.encoder(self.t_x1_tile8_r3c2)
            self.t_f_9 = self.encoder(self.t_x1_tile9_r3c3)
            self.t_f_x1_composite = tf.concat([self.t_f_1, self.t_f_2, self.t_f_3, self.t_f_4, self.t_f_5, self.t_f_6, self.t_f_7, self.t_f_8, self.t_f_9], 1)
            # create tiles for test_images_x2
            self.t_x2_tile10_r1c1 = tf.image.crop_to_bounding_box(self.test_images_x2, 0, 0, slice_size, slice_size)
            self.t_x2_tile11_r1c2 = tf.image.crop_to_bounding_box(self.test_images_x2, 0, slice_size_overlap, slice_size, slice_size)
            self.t_x2_tile12_r1c3 = tf.image.crop_to_bounding_box(self.test_images_x2, 0, image_size - slice_size, slice_size, slice_size)
            self.t_x2_tile13_r2c1 = tf.image.crop_to_bounding_box(self.test_images_x2, slice_size_overlap, 0, slice_size, slice_size)
            self.t_x2_tile14_r2c2 = tf.image.crop_to_bounding_box(self.test_images_x2, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
            self.t_x2_tile15_r2c3 = tf.image.crop_to_bounding_box(self.test_images_x2, slice_size_overlap, image_size - slice_size, slice_size, slice_size)
            self.t_x2_tile16_r3c1 = tf.image.crop_to_bounding_box(self.test_images_x2, image_size - slice_size, 0, slice_size, slice_size)
            self.t_x2_tile17_r3c2 = tf.image.crop_to_bounding_box(self.test_images_x2, image_size - slice_size, slice_size_overlap, slice_size, slice_size)
            self.t_x2_tile18_r3c3 = tf.image.crop_to_bounding_box(self.test_images_x2, image_size - slice_size, image_size - slice_size, slice_size, slice_size)
            self.t_f_10 = self.encoder(self.t_x2_tile10_r1c1)
            self.t_f_11 = self.encoder(self.t_x2_tile11_r1c2)
            self.t_f_12 = self.encoder(self.t_x2_tile12_r1c3)
            self.t_f_13 = self.encoder(self.t_x2_tile13_r2c1)
            self.t_f_14 = self.encoder(self.t_x2_tile14_r2c2)
            self.t_f_15 = self.encoder(self.t_x2_tile15_r2c3)
            self.t_f_16 = self.encoder(self.t_x2_tile16_r3c1)
            self.t_f_17 = self.encoder(self.t_x2_tile17_r3c2)
            self.t_f_18 = self.encoder(self.t_x2_tile18_r3c3)
            self.t_f_x2_composite = tf.concat([self.t_f_10, self.t_f_11, self.t_f_12, self.t_f_13, self.t_f_14, self.t_f_15, self.t_f_16, self.t_f_17, self.t_f_18], 1)

            ####################################################
            # TEST:
            # mix the features for the two test images_x1 START
            # 1. test_case: take all but 1 chunk/tile (with varying position) from the source test_x1 (test_images_x1)
            self.f_test_1_2 = tf.concat([self.t_f_10, self.t_f_2, self.t_f_3, self.t_f_4, self.t_f_5, self.t_f_6, self.t_f_7, self.t_f_8, self.t_f_9], 1)
            self.test_images_mix_one_tile = self.decoder(self.f_test_1_2, reuse=True) # used to be D_mix_allchunk
            self.test_images_mix_n_tiles = self.test_images_mix_one_tile # used to be D_mix_allchunk_sup
            for tile_id in range(1, NUM_TILES):
                self.f_test_1_2 = self.t_f_1
                for j in range(1, NUM_TILES):
                    if j == tile_id:
                        tmp = self.t_f_x2_composite[:, j * self.feature_size:(j + 1) * self.feature_size]
                        self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
                    else:
                        tmp = self.t_f_x1_composite[:, j * self.feature_size:(j + 1) * self.feature_size]
                        self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
                tmp_mix = self.decoder(self.f_test_1_2, reuse=True)
                self.test_images_mix_one_tile = tf.concat(axis=0, values=[self.test_images_mix_one_tile, tmp_mix])

            # 2. test_case: 1st tile from f_2, then increasingly with iterations: all tiles from f_2 till current tile, rest from f_1
            for tile_id in range(1, NUM_TILES):
                self.f_test_1_2 = self.t_f_10
                for j in range(1, NUM_TILES):
                    if j <= tile_id:
                        tmp = self.t_f_x2_composite[:, j * self.feature_size:(j + 1) * self.feature_size]
                        self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
                    else:
                        tmp = self.t_f_x1_composite[:, j * self.feature_size:(j + 1) * self.feature_size]
                        self.f_test_1_2 = tf.concat(axis=1, values=[self.f_test_1_2, tmp])
                tmp_mix = self.decoder(self.f_test_1_2)
                self.test_images_mix_n_tiles = tf.concat(axis=0, values=[self.test_images_mix_n_tiles, tmp_mix])
            assert self.test_images_mix_one_tile.shape == self.test_images_mix_n_tiles.shape

            # 3. test_case: use random mask to mix the two images
            for tile_id in range(0, NUM_TILES): # for each tile feature slot
                t_f_x1_tile_feature = self.t_f_x1_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                t_f_x2_tile_feature = self.t_f_x2_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                tile_mask_batchsize = tf.equal(self.mask[tile_id] * a_tile_chunk, FROM_X1)
                t_f_feature_selected = tf.where(tile_mask_batchsize, t_f_x1_tile_feature, t_f_x2_tile_feature)
                self.t_f_x1_x2_mix = t_f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.t_f_x1_x2_mix, t_f_feature_selected])
            self.test_images_mix_random = self.decoder(self.t_f_x1_x2_mix)
            assert self.t_f_x1_x2_mix.shape[0] == self.batch_size
            assert self.t_f_x1_x2_mix.shape[1] == self.feature_size * NUM_TILES
            # mix the features for the two test images_x1 END
            ##########################################################################

        with tf.variable_scope('classifier_loss'):
            # Cls loss; mask_batchsize here is GT, cls should predict correct mask..
            self.cls_loss = binary_cross_entropy_with_logits(self.mask_actual, self.mask_predicted)
            """ cls_loss: a scalar, of shape () """

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
            # this is max_D part of minmax loss function
            self.dsc_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.dsc_x3), self.dsc_x3)
            self.dsc_loss = self.dsc_loss_real + self.dsc_loss_fake
            """ dsc_loss: a scalar, of shape () """

        with tf.variable_scope('generator_loss'):
            # D (fix Dsc you have loss for G) -> cf. Dec
            # images_x3 = Dec(f_1_2) = G(f_1_2); Dsc(images_x3) = dsc_x3
            # rationale behind g_loss: this is min_G part of minmax loss function: min log D(G(x))
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_x3), self.dsc_x3)

        with tf.variable_scope('L2') as _:
            # Reconstruction loss L2 between x1 and x1' (to ensure autoencoder works properly)
            self.rec_loss_x1hat_x1 = tf.reduce_mean(tf.square(self.images_x1_hat - self.images_x1))
            """ rec_loss_x1hat_x1: a scalar, of shape () """
            # Reconstruction loss L2 between x2 and x2' (to ensure autoencoder works properly)
            self.rec_loss_x2hat_x2 = tf.reduce_mean(tf.square(self.images_x2_hat - self.images_x2))
            # L2 between x1 and x4
            self.rec_loss_x4_x1 = tf.reduce_mean(tf.square(self.images_x4 - self.images_x1))
            # L2 between x2 and x5
            self.rec_loss_x5_x2 = tf.reduce_mean(tf.square(self.images_x5 - self.images_x2))

        # TODO what for?
        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()
        # Tf stuff (tell variables how to train..)
        self.dsc_vars = [var for var in t_vars if 'd_' in var.name] # discriminator
        self.gen_vars = [var for var in t_vars if 'g_' in var.name] # encoder + decoder (generator)
        self.cls_vars = [var for var in t_vars if 'c_' in var.name] # classifier

        # save the weights
        self.saver = tf.train.Saver(self.dsc_vars + self.gen_vars + self.cls_vars + batch_norm.shadow_variables, max_to_keep=0)
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

        self.d_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        self.c_learning_rate = tf.train.exponential_decay(0.0002, global_step=global_step,
                                                          decay_steps=20000, decay_rate=0.9, staircase=True)

        labmda = 0 # should be 0
        # g_loss = labmda*self.rec_loss+10*self.recR_loss +10*self.rec_mix_loss+1*self.g_loss+1*self.cf_loss

        # entire autoencoder loss (2 Enc and 2 Dec share weights) (between x1/x2 and x4)
        # G_LOSS = composed of sub-losses from within the generator network
        # this basically refers to (4) in [1]
        # => you constrain all these losses (based on x3) on the generator network -> you can just sum them up
        # NB: lambda values: tuning trick to balance the autoencoder and the GAN
        g_loss_comp = 10 * self.rec_loss_x2hat_x2 + 10 * self.rec_loss_x4_x1 + 1 * self.g_loss + 1 * self.cls_loss
        # for autoencoder
        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=params.beta1) \
                          .minimize(g_loss_comp, var_list=self.gen_vars) # includes encoder + decoder weights
        # for classifier
        c_optim = tf.train.AdamOptimizer(learning_rate=self.c_learning_rate, beta1=params.beta1) \
                          .minimize(self.cls_loss, var_list=self.cls_vars)
        # for Dsc
        d_optim = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=params.beta1) \
                          .minimize(self.dsc_loss, var_list=self.dsc_vars, global_step=global_step)

        # what you specify in the argument to control_dependencies is ensured to be evaluated before anything you define in the with block
        with tf.control_dependencies([g_optim]):
            # this is also part of BP/training; this line is a fix re BN acc. to Stackoverflow
            g_optim = tf.group(self.bn_assigners)

        tf.global_variables_initializer().run()
        if params.continue_from:
            ckpt_name = self.load(params, params.continue_from_iteration)
            counter = int(ckpt_name[ckpt_name.rfind('-')+1:])
            print('continuing from \'%s\'...' % ckpt_name)
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
                self.sess.run([c_optim])
                self.sess.run([d_optim])
                counter += 1
                print(str(counter))

                if counter % 100 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, counter)

                if np.mod(counter, 2000) == 0:
                    # print out images every so often
                    images_x1,images_x2, images_x3,\
                    images_x1_hat,images_x2_hat,\
                    images_x4, images_x5, \
                    test_images1,test_images2, \
                    test_images_mix_one_tile,\
                    test_images_mix_n_tiles,\
                    test_images_mix_random,\
                    test_mask = \
                        self.sess.run([self.images_x1, self.images_x2, self.images_x3, \
                                       self.images_x1_hat, self.images_x2_hat, \
                                       self.images_x4, self.images_x5, \
                                       self.test_images_x1, self.test_images_x2, \
                                       self.test_images_mix_one_tile, \
                                       self.test_images_mix_n_tiles, \
                                       self.test_images_mix_random, \
                                       self.mask])

                    grid_size = np.ceil(np.sqrt(self.batch_size))
                    grid = [grid_size, grid_size]

                    save_images(images_x1,grid, self.path('%s_train_images_x1.jpg' % counter))
                    save_images(images_x2, grid, self.path('%s_train_images_x2.jpg' % counter))
                    save_images(images_x1_hat,grid, self.path('%s_train_images_x1_hat.jpg' % counter))
                    save_images(images_x2_hat, grid, self.path('%s_train_images_x2_hat.jpg' % counter))
                    save_images(images_x3, grid, self.path('%s_train_images_x3.jpg' % counter))
                    save_images(images_x4, grid, self.path('%s_train_images_x4.jpg' % counter))
                    save_images(images_x5, grid, self.path('%s_train_images_x5.jpg' % counter))

                    save_images(test_images1, grid, self.path('%s_test_images_x1.jpg' % counter))
                    save_images(test_images2, grid, self.path('%s_test_images_x2.jpg' % counter))
                    save_images_one_every_batch(test_images_mix_one_tile, grid, self.batch_size, self.path('%s_test_images_mix_one_tile.jpg' % counter))
                    save_images_one_every_batch(test_images_mix_n_tiles, grid, self.batch_size, self.path('%s_test_images_mix_n_tiles.jpg' % counter))

                    grid_test = [self.batch_size, NUM_TILES+2]
                    file_path = self.path('%s_test_mix_one_tile_comparison.jpg' % counter)
                    save_images_multi(test_images1, test_images2, test_images_mix_one_tile, grid_test, self.batch_size, file_path)
                    file_path = self.path('%s_test_mix_n_tiles_comparison.jpg' % counter)
                    save_images_multi(test_images1, test_images2, test_images_mix_n_tiles, grid_test, self.batch_size, file_path)
                    file_path = self.path('%s_test_mix_random_%s.jpg' % (counter, ''.join(str(e) for e in test_mask)))
                    grid_test = [self.batch_size, 3]
                    save_images_multi(test_images1, test_images2, test_images_mix_random, grid_test, self.batch_size, file_path)

                if np.mod(counter, 600) == 1:
                    self.save(params.checkpoint_dir, counter)


        except Exception as e:
            if 'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Error during training:')
                print(e)
            if counter > 0:
                self.save(params.checkpoint_dir, counter) # save model again
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
        h0 = lrelu(self.d_bn1(conv2d(image, self.df_dim, name='d_1_h0_conv')))
        h1 = lrelu(self.d_bn2(conv2d(h0, self.df_dim*2, name='d_1_h1_conv')))
        h2 = lrelu(self.d_bn3(conv2d(h1, self.df_dim*4, name='d_1_h2_conv')))
        # NB: k=1,d=1 is like an FC layer -> to strengthen h3, to give it more capacity
        h3 = lrelu(self.d_bn4(conv2d(h2, self.df_dim*8,k_h=1, k_w=1, d_h=1, d_w=1, name='d_1_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_1_h3_lin')

        return tf.nn.sigmoid(h4)


    def classifier(self, x1_tile1, x1_tile2, x1_tile3, x1_tile4, x1_tile5, x1_tile6, x1_tile7, x1_tile8, x1_tile9,
                   x2_tile10, x2_tile11, x2_tile12, x2_tile13, x2_tile14, x2_tile15, x2_tile16, x2_tile17, x2_tile18,
                   x3_tile1, x3_tile2, x3_tile3, x3_tile4, x3_tile5, x3_tile6, x3_tile7, x3_tile8, x3_tile9,
                   reuse=False):
        """From paper:
        For the classifier, we use AlexNet with batch normalization after each
        convolutional layer, but we do not use any dropout. The image inputs of
        the classifier are concatenated along the RGB channels.

        returns: a 1D matrix of size NUM_TILES i.e. (batch_size, 9)
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        concatenated = tf.concat(axis=3, values=[x1_tile1, x1_tile2, x1_tile3, x1_tile4, x1_tile5, x1_tile6, x1_tile7, x1_tile8, x1_tile9])
        concatenated = tf.concat(axis=3, values=[concatenated, x2_tile10, x2_tile11, x2_tile12, x2_tile13, x2_tile14, x2_tile15, x2_tile16, x2_tile17, x2_tile18])
        concatenated = tf.concat(axis=3, values=[concatenated, x3_tile1, x3_tile2, x3_tile3, x3_tile4, x3_tile5, x3_tile6, x3_tile7, x3_tile8, x3_tile9])
        """ concatenated is of size (batch_size, 110, 110, 81) """

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

        self.fc8 = linear(tf.reshape(fc7, [self.batch_size, -1]), NUM_TILES, 'c_3_fc8')

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


    def decoder(self, representations, reuse=False):
        """
        returns: batch of images with size 256x60x60x3
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshape = tf.reshape(representations,[self.batch_size, 1, 1, NUM_TILES * self.feature_size])
        # TODO consider increasing capacity of decoder since feature_size-dim is NUM_TILES bigger...
        h = deconv2d(reshape, [self.batch_size, 5, 5, self.gf_dim*4], k_h=5, k_w=5, d_h=1, d_w=1, padding='VALID', name='g_de_h')
        h = tf.nn.relu(h)

        h1 = deconv2d(h, [self.batch_size, 10, 10, self.gf_dim*4 ], name='g_h1')
        h1 = tf.nn.relu(instance_norm(h1))

        h2 = deconv2d(h1, [self.batch_size, 19, 19, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(instance_norm(h2))

        h3 = deconv2d(h2, [self.batch_size, 38, 38, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(instance_norm(h3))

        h4 = deconv2d(h3, [self.batch_size, 75, 75, self.c_dim], name='g_h4')
        h4 = tf.nn.relu(instance_norm(h4))

        h5 = deconv2d(h4, [self.batch_size, 150, 150, self.c_dim], name='g_h5')
        h5 = tf.nn.relu(instance_norm(h5))

        h6 = deconv2d(h5, [self.batch_size, 300, 300, self.c_dim], name='g_h6')

        return tf.nn.tanh(h6)


    def make_summary_ops(self, g_loss_comp):
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('g_loss_comp', g_loss_comp)
        tf.summary.scalar('cls_loss', self.cls_loss)
        tf.summary.scalar('dsc_loss', self.dsc_loss)
        tf.summary.scalar('dsc_loss_fake', self.dsc_loss_fake)
        tf.summary.scalar('dsc_loss_real', self.dsc_loss_real)
        tf.summary.scalar('rec_loss_x1hat_x1', self.rec_loss_x1hat_x1)
        tf.summary.scalar('rec_loss_x2hat_x2', self.rec_loss_x2hat_x2)
        tf.summary.scalar('rec_loss_x4_x1', self.rec_loss_x4_x1)
        tf.summary.scalar('rec_loss_x5_x2', self.rec_loss_x5_x2)


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
