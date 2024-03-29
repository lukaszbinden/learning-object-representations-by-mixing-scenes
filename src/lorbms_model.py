import signal
from ops_alex import *
from ops_coordconv import *
from utils_dcgan import *
from utils_common import *
from input_pipeline import *
# from tensorflow.contrib.receptive_field import receptive_field_api as receptive_field
from autoencoder_dblocks import encoder_dense, decoder_dense
from patch_gan_discriminator import Deep_PatchGAN_Discrminator
from sbd import decoder_sbd
from constants import *
import numpy as np
from scipy.misc import imsave
import traceback
import csv
from random import randint

class DCGAN(object):

    def __init__(self, sess, params,
                 batch_size=256, sample_size = 64, epochs=1000, image_shape=[256, 256, 3],
                 y_dim=None, z_dim=0, gf_dim=128, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1,
                 is_train=True, random_seed=4285):
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

        # self.g_s_bn5 = batch_norm(is_train,convolutional=False, name='g_s_bn5')

        self.end = False

        self.random_seed = random_seed

        # exp74:
        self.useIRefAndMixForGanLoss = False
        self.lambda_mix = 0.85
        self.lambda_ref = 0.15
        # exp77:
        self.useNoiseInDecoder = False

        self.build_model()


    def build_model(self):
        print("build_model() ------------------------------------------>")
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        tf.set_random_seed(self.random_seed)

        image_size = self.image_size

        isIdeRun = 'lz826' in os.path.realpath(sys.argv[0])
        file_train = self.params.tfrecords_path if not isIdeRun else 'data/train-00011-of-00060.tfrecords'

        ####################################################################################
        reader = tf.TFRecordReader()
        rrm_fn = lambda name : read_record_max(name, reader, image_size)
        filenames, train_images, t1_10nn_ids, t1_10nn_subids, t1_10nn_L2, t2_10nn_ids, t2_10nn_subids, t2_10nn_L2, t3_10nn_ids, t3_10nn_subids, t3_10nn_L2, t4_10nn_ids, t4_10nn_subids, t4_10nn_L2 = \
                get_pipeline(file_train, self.batch_size, self.epochs, rrm_fn)
        print('train_images.shape..:', train_images.shape)
        self.fnames_I_ref = filenames
        self.images_I_ref = train_images

        # create tiles for I_ref (only for logging purposes)
        tile_size = image_size / 2
        assert tile_size.is_integer()
        tile_size = int(tile_size)
        self.I_ref_t1 = tf.image.crop_to_bounding_box(self.images_I_ref, 0, 0, tile_size, tile_size)
        self.I_ref_t2 = tf.image.crop_to_bounding_box(self.images_I_ref, 0, tile_size, tile_size, tile_size)
        self.I_ref_t3 = tf.image.crop_to_bounding_box(self.images_I_ref, tile_size, 0, tile_size, tile_size)
        self.I_ref_t4 = tf.image.crop_to_bounding_box(self.images_I_ref, tile_size, tile_size, tile_size, tile_size)

        t1_10nn_ids = tf.reshape(tf.sparse.to_dense(t1_10nn_ids), (self.batch_size, -1))
        t2_10nn_ids = tf.reshape(tf.sparse.to_dense(t2_10nn_ids), (self.batch_size, -1))
        t3_10nn_ids = tf.reshape(tf.sparse.to_dense(t3_10nn_ids), (self.batch_size, -1))
        t4_10nn_ids = tf.reshape(tf.sparse.to_dense(t4_10nn_ids), (self.batch_size, -1))

        t1_10nn_subids = tf.reshape(tf.sparse.to_dense(t1_10nn_subids), (self.batch_size, -1))
        t2_10nn_subids = tf.reshape(tf.sparse.to_dense(t2_10nn_subids), (self.batch_size, -1))
        t3_10nn_subids = tf.reshape(tf.sparse.to_dense(t3_10nn_subids), (self.batch_size, -1))
        t4_10nn_subids = tf.reshape(tf.sparse.to_dense(t4_10nn_subids), (self.batch_size, -1))

        t1_10nn_L2 = tf.reshape(tf.sparse.to_dense(t1_10nn_L2), (self.batch_size, -1))
        t2_10nn_L2 = tf.reshape(tf.sparse.to_dense(t2_10nn_L2), (self.batch_size, -1))
        t3_10nn_L2 = tf.reshape(tf.sparse.to_dense(t3_10nn_L2), (self.batch_size, -1))
        t4_10nn_L2 = tf.reshape(tf.sparse.to_dense(t4_10nn_L2), (self.batch_size, -1))

        nn_id = tf.random_uniform([self.batch_size], 0, 9, dtype=tf.int32, seed=4285)

        path = self.params.full_imgs_path if not isIdeRun else 'D:\\learning-object-representations-by-mixing-scenes\\src\\datasets\\coco\\2017_training\\version\\v4\\full\\'
        path = tf.constant(path)
        filetype = tf.constant(".jpg")

        # kNN images of quadrant t1 ############################################################################################
        # path_prefix_t1 = path + tf.constant("/t1/")

        for id in range(self.batch_size):
            t1_10nn_ids_b = t1_10nn_ids[id]
            index = nn_id[id]
            t1_10nn_id = tf.gather(t1_10nn_ids_b, index)
            t1_10nn_id_str = tf.as_string(t1_10nn_id)
            t1_10nn_subids_b = t1_10nn_subids[id]
            t1_10nn_subid = tf.gather(t1_10nn_subids_b, index)
            t1_10nn_subid_str = tf.as_string(t1_10nn_subid)
            postfix = underscore + t1_10nn_subid_str + filetype
            fname = get_coco_filename(t1_10nn_id_str, postfix)
            t1_10nn_fnames = fname if id == 0 else tf.concat(axis=0, values=[t1_10nn_fnames, fname])
        self.images_t1_fnames = t1_10nn_fnames
        with tf.control_dependencies([tf.assert_equal(self.batch_size, t1_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t1_10nn_fnames), 18)]):
            t1_10nn_fnames = tf.strings.join([path, t1_10nn_fnames])

            for id in range(self.batch_size):
                file = tf.read_file(t1_10nn_fnames[id])
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, image_size, self.batch_size)
                file = tf.expand_dims(file, 0)
                t1_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t1_10nn_images, file])
        self.images_t1 = t1_10nn_images
        # create tile for I_t1 (only for logging purposes)
        self.I_t1_tile = tf.image.crop_to_bounding_box(self.images_t1, 0, 0, tile_size, tile_size)


        # kNN images of quadrant t2 ############################################################################################
        for id in range(self.batch_size):
            t2_10nn_ids_b = t2_10nn_ids[id]
            index = nn_id[id]
            t2_10nn_id = tf.gather(t2_10nn_ids_b, index)
            t2_10nn_id_str = tf.as_string(t2_10nn_id)
            t2_10nn_subids_b = t2_10nn_subids[id]
            t2_10nn_subid = tf.gather(t2_10nn_subids_b, index)
            t2_10nn_subid_str = tf.as_string(t2_10nn_subid)
            postfix = underscore + t2_10nn_subid_str + filetype
            fname = get_coco_filename(t2_10nn_id_str, postfix)
            t2_10nn_fnames = fname if id == 0 else tf.concat(axis=0, values=[t2_10nn_fnames, fname])
        self.images_t2_fnames = t2_10nn_fnames
        with tf.control_dependencies([tf.assert_equal(self.batch_size, t2_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t2_10nn_fnames), 18)]):
            t2_10nn_fnames = tf.strings.join([path, t2_10nn_fnames])
            for id in range(self.batch_size):
                file = tf.read_file(t2_10nn_fnames[id])
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, image_size, self.batch_size)
                file = tf.expand_dims(file, 0)
                t2_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t2_10nn_images, file])
        self.images_t2 = t2_10nn_images
        # create tile for I_t2 (only for logging purposes)
        self.I_t2_tile = tf.image.crop_to_bounding_box(self.images_t2, 0, tile_size, tile_size, tile_size)


        # kNN images of quadrant t3 ############################################################################################
        for id in range(self.batch_size):
            t3_10nn_ids_b = t3_10nn_ids[id]
            index = nn_id[id]
            t3_10nn_id = tf.gather(t3_10nn_ids_b, index)
            t3_10nn_id_str = tf.as_string(t3_10nn_id)
            t3_10nn_subids_b = t3_10nn_subids[id]
            t3_10nn_subid = tf.gather(t3_10nn_subids_b, index)
            t3_10nn_subid_str = tf.as_string(t3_10nn_subid)
            postfix = underscore + t3_10nn_subid_str + filetype
            fname = get_coco_filename(t3_10nn_id_str, postfix)
            t3_10nn_fnames = fname if id == 0 else tf.concat(axis=0, values=[t3_10nn_fnames, fname])
        self.images_t3_fnames = t3_10nn_fnames
        with tf.control_dependencies([tf.assert_equal(self.batch_size, t3_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t3_10nn_fnames), 18)]):
            t3_10nn_fnames = tf.strings.join([path, t3_10nn_fnames])
            for id in range(self.batch_size):
                file = tf.read_file(t3_10nn_fnames[id])
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, image_size, self.batch_size)
                file = tf.expand_dims(file, 0)
                t3_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t3_10nn_images, file])
        self.images_t3 = t3_10nn_images
        # create tile for I_t3 (only for logging purposes)
        self.I_t3_tile = tf.image.crop_to_bounding_box(self.images_t3, tile_size, 0, tile_size, tile_size)


        # kNN images of quadrant t4 ############################################################################################
        for id in range(self.batch_size):
            t4_10nn_ids_b = t4_10nn_ids[id]
            index = nn_id[id]
            t4_10nn_id = tf.gather(t4_10nn_ids_b, index)
            t4_10nn_id_str = tf.as_string(t4_10nn_id)
            t4_10nn_subids_b = t4_10nn_subids[id]
            t4_10nn_subid = tf.gather(t4_10nn_subids_b, index)
            t4_10nn_subid_str = tf.as_string(t4_10nn_subid)
            postfix = underscore + t4_10nn_subid_str + filetype
            fname = get_coco_filename(t4_10nn_id_str, postfix)
            t4_10nn_fnames = fname if id == 0 else tf.concat(axis=0, values=[t4_10nn_fnames, fname])
        self.images_t4_fnames = t4_10nn_fnames
        with tf.control_dependencies([tf.assert_equal(self.batch_size, t4_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t4_10nn_fnames), 18)]):
            t4_10nn_fnames = tf.strings.join([path, t4_10nn_fnames])
            for id in range(self.batch_size):
                file = tf.read_file(t4_10nn_fnames[id])
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, image_size, self.batch_size)
                file = tf.expand_dims(file, 0)
                t4_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t4_10nn_images, file])
        self.images_t4 = t4_10nn_images
        # create tile for I_t4 (only for logging purposes)
        self.I_t4_tile = tf.image.crop_to_bounding_box(self.images_t4, tile_size, tile_size, tile_size, tile_size)


        # ###########################################################################################################
        # ###########################################################################################################

        # 12.11: currently leave scaling idea out and first focus on the core clustering idea

        self.chunk_num = self.params.chunk_num
        """ number of chunks: 8 """
        self.chunk_size = self.params.chunk_size
        """ size per chunk: 64 """
        self.feature_size_tile = self.chunk_size * self.chunk_num
        """ equals the size of all chunks from a single tile """
        self.feature_size = self.feature_size_tile * NUM_TILES_L2_MIX
        """ equals the size of the full image feature """

        # each tile chunk is initialized with 1's
        a_tile_chunk = tf.ones((self.batch_size, self.feature_size_tile), dtype=tf.int32)
        assert a_tile_chunk.shape[0] == self.batch_size
        assert a_tile_chunk.shape[1] == self.feature_size_tile

        with tf.variable_scope('generator') as scope_generator:
            #self.I_ref_f1 = self.encoder(self.I_ref_t1)
            # params for ENCODER
            model = self.params.autoencoder_model
            coordConvLayer = True
            ####################

            self.I_ref_f = encoder_dense(self.images_I_ref, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)
            assert self.I_ref_f.shape[0] == self.batch_size
            assert self.I_ref_f.shape[1] == self.feature_size

            # if model == 'FC-DenseNet-RF-46':
            #     (receptive_field_x, receptive_field_y, _, _, _, _) = receptive_field.compute_receptive_field_from_graph_def(
            #         self.sess.graph, "generator/g_1_enc/first_conv/Conv2D", "generator/g_1_enc/transitiondown-final/max_pool")
            #     assert receptive_field_x == receptive_field_y
            #     print('receptive field: %dx%d' % (receptive_field_x, receptive_field_y))
            # elif model == 'encoder_rf46':
            #     (receptive_field_x, receptive_field_y, _, _, _, _) = receptive_field.compute_receptive_field_from_graph_def(
            #         self.sess.graph, "generator/g_1_enc/first_conv/Conv2D", "generator/g_1_enc/TD-final/TD-final_2_co/BiasAdd")
            #     assert receptive_field_x == receptive_field_y
            #     print('receptive field: %dx%d' % (receptive_field_x, receptive_field_y))
            #     assert 1 == 0
            # else:
            #     (receptive_field_x, receptive_field_y, _, _, _, _) = receptive_field.compute_receptive_field_from_graph_def(
            #         self.sess.graph, "generator/g_1_enc/first_conv/Conv2D", "generator/g_1_enc/logits/BiasAdd")
            #     assert receptive_field_x == receptive_field_y
            #     print('receptive field: %dx%d' % (receptive_field_x, receptive_field_y))

            feature_tile_shape = [self.batch_size, self.feature_size_tile]
            self.I_ref_f1 = tf.slice(self.I_ref_f, [0, self.feature_size_tile * 0], feature_tile_shape)
            self.I_ref_f2 = tf.slice(self.I_ref_f, [0, self.feature_size_tile * 1], feature_tile_shape)
            self.I_ref_f3 = tf.slice(self.I_ref_f, [0, self.feature_size_tile * 2], feature_tile_shape)
            self.I_ref_f4 = tf.slice(self.I_ref_f, [0, self.feature_size_tile * 3], feature_tile_shape)
            assert self.I_ref_f1.shape[0] == self.batch_size
            assert self.I_ref_f1.shape[1] == self.feature_size_tile
            assert self.I_ref_f1.shape == self.I_ref_f2.shape
            assert self.I_ref_f3.shape == self.I_ref_f4.shape

            # this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.decoder(self.I_ref_f, preset_model=model, dropout_p=0.0)

            # Classifier
            # -> this is used to build up graph nodes (variables) -> for later reuse_variables..
            #__self.classifier(self.images_I_ref, self.images_I_ref, self.images_I_ref, self.images_I_ref, self.images_I_ref, self.images_I_ref)
            self.classifier_two_image(self.images_I_ref, self.images_I_ref)

            # to share the weights between the Encoders
            scope_generator.reuse_variables()

            self.I_t1_f = encoder_dense(self.images_t1, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)
            self.t1_f = tf.slice(self.I_t1_f, [0, self.feature_size_tile * 0], feature_tile_shape)
            self.I_t2_f = encoder_dense(self.images_t2, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)
            self.t2_f = tf.slice(self.I_t2_f, [0, self.feature_size_tile * 1], feature_tile_shape)
            self.I_t3_f = encoder_dense(self.images_t3, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)
            self.t3_f = tf.slice(self.I_t3_f, [0, self.feature_size_tile * 2], feature_tile_shape)
            self.I_t4_f = encoder_dense(self.images_t4, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)
            self.t4_f = tf.slice(self.I_t4_f, [0, self.feature_size_tile * 3], feature_tile_shape)

            # ###########################################################################################################
            # 1) replace tile w/ max L2 wrt I_ref w/ respective tile of I_ref
            # 2) replaces tiles t_i w/ I_ref where L2(t_i) > tau
            # 3) ensure tile t_i w/ min L2(t_i) is selected
            # ultimately, we want to construct f_Iref_I2_mix for generation of new image
            tau = self.params.threshold_L2

            for id in range(self.batch_size):
                index = nn_id[id]

                t1_10nn_L2_b = tf.gather(t1_10nn_L2[id], index)
                t2_10nn_L2_b = tf.gather(t2_10nn_L2[id], index)
                t3_10nn_L2_b = tf.gather(t3_10nn_L2[id], index)
                t4_10nn_L2_b = tf.gather(t4_10nn_L2[id], index)
                all_L2 = tf.stack(axis=0, values=[t1_10nn_L2_b, t2_10nn_L2_b, t3_10nn_L2_b, t4_10nn_L2_b])
                argmax_L2 = tf.argmax(tf.reshape(all_L2, [-1]), axis=0)
                argmin_L2 = tf.argmin(tf.reshape(all_L2, [-1]), axis=0)

                # pick I_ref_t1 IFF t1 is argmax L2 or L2 > TAU and t1 is not argmin L2
                is_t1_maxL2 = tf.equal(argmax_L2, 0)
                is_t1_minL2 = tf.equal(argmin_L2, 0)
                cond_Iref_t1 = tf.logical_and(tf.logical_or(is_t1_maxL2, tf.greater(t1_10nn_L2_b, tau)), tf.logical_not(is_t1_minL2))
                tile_1 = tf.expand_dims(tf.where(cond_Iref_t1, self.I_ref_t1[id], self.I_t1_tile[id]), 0)
                # for the assignment mask e.g. [0 1 1 0], of shape (4,)
                # 0 selects the corresponding tile from I_ref
                # 1 selects the corresponding tile from I_M
                assignment_1 = tf.where(cond_Iref_t1, 0, 1)
                self.J_1_tile = tile_1 if id == 0 else tf.concat(axis=0, values=[self.J_1_tile, tile_1])
                feature_1 = tf.expand_dims(tf.where(cond_Iref_t1, self.I_ref_f1[id], self.t1_f[id]), 0)

                is_t2_maxL2 = tf.equal(argmax_L2, 1)
                is_t2_minL2 = tf.equal(argmin_L2, 1)
                cond_Iref_t2 = tf.logical_and(tf.logical_or(is_t2_maxL2, tf.greater(t2_10nn_L2_b, tau)), tf.logical_not(is_t2_minL2))
                tile_2 = tf.expand_dims(tf.where(cond_Iref_t2, self.I_ref_t2[id], self.I_t2_tile[id]), 0)
                assignment_2 = tf.where(cond_Iref_t2, 0, 1)
                self.J_2_tile = tile_2 if id == 0 else tf.concat(axis=0, values=[self.J_2_tile, tile_2])
                feature_2 = tf.expand_dims(tf.where(cond_Iref_t2, self.I_ref_f2[id], self.t2_f[id]), 0)

                is_t3_maxL2 = tf.equal(argmax_L2, 2)
                is_t3_minL2 = tf.equal(argmin_L2, 2)
                cond_Iref_t3 = tf.logical_and(tf.logical_or(is_t3_maxL2, tf.greater(t3_10nn_L2_b, tau)), tf.logical_not(is_t3_minL2))
                tile_3 = tf.expand_dims(tf.where(cond_Iref_t3, self.I_ref_t3[id], self.I_t3_tile[id]), 0)
                assignment_3 = tf.where(cond_Iref_t3, 0, 1)
                self.J_3_tile = tile_3 if id == 0 else tf.concat(axis=0, values=[self.J_3_tile, tile_3])
                feature_3 = tf.expand_dims(tf.where(cond_Iref_t3, self.I_ref_f3[id], self.t3_f[id]), 0)

                is_t4_maxL2 = tf.equal(argmax_L2, 3)
                is_t4_minL2 = tf.equal(argmin_L2, 3)
                cond_Iref_t4 = tf.logical_and(tf.logical_or(is_t4_maxL2, tf.greater(t4_10nn_L2_b, tau)), tf.logical_not(is_t4_minL2))
                tile_4 = tf.expand_dims(tf.where(cond_Iref_t4, self.I_ref_t4[id], self.I_t4_tile[id]), 0)
                assignment_4 = tf.where(cond_Iref_t4, 0, 1)
                self.J_4_tile = tile_4 if id == 0 else tf.concat(axis=0, values=[self.J_4_tile, tile_4])
                feature_4 = tf.expand_dims(tf.where(cond_Iref_t4, self.I_ref_f4[id], self.t4_f[id]), 0)

                # only for logging purposes START
                assignments = tf.stack(axis=0, values=[assignment_1, assignment_2, assignment_3, assignment_4])
                assignments = tf.expand_dims(tf.reshape(assignments, [-1]), 0)
                self.assignments_actual = assignments if id == 0 else tf.concat(axis=0, values=[self.assignments_actual, assignments])  # or 'mask'
                # only for logging purposes END

                next_assignment = tf.stack(axis=0, values=[assignment_1, ZERO, ZERO, ZERO])
                next_assignment = tf.expand_dims(tf.reshape(next_assignment, [-1]), 0)
                self.assignments_actual_t1 = next_assignment if id == 0 else tf.concat(axis=0, values=[self.assignments_actual_t1, next_assignment])
                next_assignment = tf.stack(axis=0, values=[ZERO, assignment_2, ZERO, ZERO])
                next_assignment = tf.expand_dims(tf.reshape(next_assignment, [-1]), 0)
                self.assignments_actual_t2 = next_assignment if id == 0 else tf.concat(axis=0, values=[self.assignments_actual_t2, next_assignment])
                next_assignment = tf.stack(axis=0, values=[ZERO, ZERO, assignment_3, ZERO])
                next_assignment = tf.expand_dims(tf.reshape(next_assignment, [-1]), 0)
                self.assignments_actual_t3 = next_assignment if id == 0 else tf.concat(axis=0, values=[self.assignments_actual_t3, next_assignment])
                next_assignment = tf.stack(axis=0, values=[ZERO, ZERO, ZERO, assignment_4])
                next_assignment = tf.expand_dims(tf.reshape(next_assignment, [-1]), 0)
                self.assignments_actual_t4 = next_assignment if id == 0 else tf.concat(axis=0, values=[self.assignments_actual_t4, next_assignment])

                assert feature_1.shape[0] == 1
                assert feature_1.shape[1] == self.feature_size_tile
                assert feature_1.shape[0] == feature_2.shape[0] and feature_1.shape[1] == feature_2.shape[1]
                assert feature_2.shape[0] == feature_3.shape[0] and feature_2.shape[1] == feature_3.shape[1]
                assert feature_2.shape[0] == feature_4.shape[0] and feature_2.shape[1] == feature_4.shape[1]
                assert feature_1.shape[1] == a_tile_chunk.shape[1]

                f_features_selected = tf.concat(axis=0, values=[feature_1, feature_2, feature_3, feature_4])  # axis=1
                f_features_selected = tf.reshape(f_features_selected, [-1])
                f_features_selected = tf.expand_dims(f_features_selected, 0)
                self.f_I_ref_I_M_mix = f_features_selected if id == 0 else tf.concat(axis=0, values=[self.f_I_ref_I_M_mix, f_features_selected])

            assert self.assignments_actual_t1.shape[0] == self.batch_size
            assert self.assignments_actual_t1.shape[1] == NUM_TILES_L2_MIX
            assert self.assignments_actual_t1.shape == self.assignments_actual_t2.shape
            assert self.assignments_actual_t2.shape == self.assignments_actual_t3.shape
            assert self.assignments_actual_t3.shape == self.assignments_actual_t4.shape
            assert self.f_I_ref_I_M_mix.shape[0] == self.batch_size
            assert self.f_I_ref_I_M_mix.shape[1] == self.feature_size

            # just for logging purposes __start ###
            row1 = tf.concat([self.J_1_tile, self.J_3_tile], axis=1)
            row2 = tf.concat([self.J_2_tile, self.J_4_tile], axis=1)
            self.images_I_M_mix = tf.concat([row1, row2], axis=2)
            # just for logging purposes __end ###

            # build composite feature including all I_ref tile features
            self.images_I_ref_hat = self.decoder(self.I_ref_f, preset_model=model, dropout_p=0.0)
            assert self.images_I_ref_hat.shape[1] == self.image_size
            # Enc/Dec for I_ref __end ##########################################

            self.images_t1_hat = self.decoder(self.I_t1_f, preset_model=model, dropout_p=0.0)

            # Dec I_ref_I_M_mix
            self.images_I_ref_I_M_mix = self.decoder(self.f_I_ref_I_M_mix, preset_model=model, dropout_p=0.0)

            # CLS
            #__ self.assignments_predicted = self.classifier(self.images_I_ref_I_M_mix, self.images_I_ref, self.images_t1, self.images_t2, self.images_t3, self.images_t4)
            self.assignments_predicted_t1 = self.classifier_two_image(self.images_I_ref_I_M_mix, self.images_t1)
            self.assignments_predicted_t2 = self.classifier_two_image(self.images_I_ref_I_M_mix, self.images_t2)
            self.assignments_predicted_t3 = self.classifier_two_image(self.images_I_ref_I_M_mix, self.images_t3)
            self.assignments_predicted_t4 = self.classifier_two_image(self.images_I_ref_I_M_mix, self.images_t4)

            """ assignments_predicted is of size (batch_size, 4) """
            assert self.assignments_predicted_t1.shape[0] == self.batch_size
            assert self.assignments_predicted_t1.shape[1] == NUM_TILES_L2_MIX
            assert self.assignments_predicted_t1.shape == self.assignments_predicted_t2.shape
            assert self.assignments_predicted_t2.shape == self.assignments_predicted_t3.shape
            assert self.assignments_predicted_t3.shape == self.assignments_predicted_t4.shape

            # # cf original mask
            # self.mask_actual = tf.cast(tf.ones((self.batch_size, NUM_TILES_L2_MIX), dtype=tf.int32) * self.mask, tf.float32)
            # """ mask_actual: mask (4,) scaled to batch_size, of shape (64, 4) """
            assert self.assignments_predicted_t1.shape == self.assignments_actual_t1.shape

            # build composite feature including all I1 tile features
            self.f_I_ref_I_M_mix_hat = encoder_dense(self.images_I_ref_I_M_mix, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)
            assert self.f_I_ref_I_M_mix_hat.shape == self.f_I_ref_I_M_mix.shape
            assert self.f_I_ref_I_M_mix_hat.shape[1] == self.feature_size


            # RECONSTRUCT I_ref_f_hat/I_t1_f_hat etc. FROM f_I_ref_I_M_mix_hat START
            # reconstruction of feature vector for tile t1
            tile_id = 0
            f_mix_tile_feature, tile_assignments = self.reconstruct_I_ref_f(tile_id, a_tile_chunk)
            I_t1_f_tile1 = tf.where(tf.equal(tile_assignments * a_tile_chunk, FROM_I_M), f_mix_tile_feature, self.t1_f)
            assert I_t1_f_tile1.shape == a_tile_chunk.shape
            I_t1_f_post = tf.slice(self.I_t1_f, [0, self.feature_size_tile * 1], [self.batch_size, self.feature_size_tile * 3])
            self.I_t1_f_hat = tf.concat(axis=1, values=[I_t1_f_tile1, I_t1_f_post])
            assert self.I_t1_f_hat.shape ==  self.I_t1_f.shape

            # reconstruction of feature vector for tile t2
            tile_id = 1
            f_mix_tile_feature, tile_assignments = self.reconstruct_I_ref_f(tile_id, a_tile_chunk)
            I_t2_f_tile2 = tf.where(tf.equal(tile_assignments * a_tile_chunk, FROM_I_M), f_mix_tile_feature, self.t2_f)
            assert I_t2_f_tile2.shape == a_tile_chunk.shape
            I_t2_f_pre = tf.slice(self.I_t2_f, [0, self.feature_size_tile * 0], feature_tile_shape)
            I_t2_f_post = tf.slice(self.I_t2_f, [0, self.feature_size_tile * 2], [self.batch_size, self.feature_size_tile * 2])
            self.I_t2_f_hat = tf.concat(axis=1, values=[I_t2_f_pre, I_t2_f_tile2, I_t2_f_post])
            assert self.I_t2_f_hat.shape == self.I_t2_f.shape

            # reconstruction of feature vector for tile t3
            tile_id = 2
            f_mix_tile_feature, tile_assignments = self.reconstruct_I_ref_f(tile_id, a_tile_chunk)
            I_t3_f_tile3 = tf.where(tf.equal(tile_assignments * a_tile_chunk, FROM_I_M), f_mix_tile_feature, self.t3_f)
            assert I_t3_f_tile3.shape == a_tile_chunk.shape
            assert self.I_t3_f.shape[1] == self.feature_size_tile * 4
            I_t3_f_pre = tf.slice(self.I_t3_f, [0, self.feature_size_tile * 0], [self.batch_size, self.feature_size_tile * 2])
            I_t3_f_post = tf.slice(self.I_t3_f, [0, self.feature_size_tile * 3], feature_tile_shape)
            self.I_t3_f_hat = tf.concat(axis=1, values=[I_t3_f_pre, I_t3_f_tile3, I_t3_f_post])
            assert self.I_t3_f_hat.shape == self.I_t3_f.shape

            # reconstruction of feature vector for tile t4
            tile_id = 3
            f_mix_tile_feature, tile_assignments = self.reconstruct_I_ref_f(tile_id, a_tile_chunk)
            I_t4_f_tile4 = tf.where(tf.equal(tile_assignments * a_tile_chunk, FROM_I_M), f_mix_tile_feature, self.t4_f)
            assert I_t4_f_tile4.shape == a_tile_chunk.shape
            I_t4_f_pre = tf.slice(self.I_t4_f, [0, self.feature_size_tile * 0], [self.batch_size, self.feature_size_tile * 3])
            self.I_t4_f_hat = tf.concat(axis=1, values=[I_t4_f_pre, I_t4_f_tile4])
            assert self.I_t4_f_hat.shape == self.I_t4_f.shape

            assert self.I_ref_f_hat.shape[0] == self.batch_size
            assert self.I_ref_f_hat.shape[1] == self.feature_size
            assert self.I_ref_f_hat.shape == self.I_ref_f.shape
            # RECONSTRUCT I_ref_f_hat/I_t1_f_hat etc. FROM f_I_ref_I_M_mix_hat END

            # decode to I_ref_4 for L2 with I_ref
            self.images_I_ref_4 = self.decoder(self.I_ref_f_hat, preset_model=model, dropout_p=0.0)
            """ images_I4: batch of reconstructed images I4 with shape (batch_size, 128, 128, 3) """
            # decode to t1_4 for L2 with t1
            self.images_t1_4 = self.decoder(self.I_t1_f_hat, preset_model=model, dropout_p=0.0)
            self.images_t2_4 = self.decoder(self.I_t2_f_hat, preset_model=model, dropout_p=0.0)
            self.images_t3_4 = self.decoder(self.I_t3_f_hat, preset_model=model, dropout_p=0.0)
            self.images_t4_4 = self.decoder(self.I_t4_f_hat, preset_model=model, dropout_p=0.0)

            self.images_I_ref_hat_psnr = tf.reduce_mean(tf.image.psnr(self.images_I_ref, self.images_I_ref_hat, max_val=1.0))
            self.images_I_ref_4_psnr = tf.reduce_mean(tf.image.psnr(self.images_I_ref, self.images_I_ref_4, max_val=1.0))
            self.images_t1_4_psnr = tf.reduce_mean(tf.image.psnr(self.images_t1, self.images_t1_4, max_val=1.0))
            self.images_t3_4_psnr = tf.reduce_mean(tf.image.psnr(self.images_t3, self.images_t3_4, max_val=1.0))


        with tf.variable_scope('classifier_loss'):
            # Cls loss; assignments_actual here is GT, cls should predict correct mask..
            cls_loss_t1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.assignments_predicted_t1, labels=tf.cast(self.assignments_actual_t1, tf.float32)))
            cls_loss_t2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.assignments_predicted_t2, labels=tf.cast(self.assignments_actual_t2, tf.float32)))
            cls_loss_t3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.assignments_predicted_t3, labels=tf.cast(self.assignments_actual_t3, tf.float32)))
            cls_loss_t4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.assignments_predicted_t4, labels=tf.cast(self.assignments_actual_t4, tf.float32)))
            self.cls_loss = 0.25 * cls_loss_t1 + 0.25 * cls_loss_t2 +  0.25 * cls_loss_t3 +  0.25 * cls_loss_t4

            """ cls_loss: a scalar, of shape () """

        with tf.variable_scope('discriminator'):
            # Dsc for I1
            self.dsc_I_ref = self.discriminator(self.images_I_ref)
            """ dsc_I_ref: real/fake, of shape (64, 1) """
            # Dsc for I3
            self.dsc_I_ref_I_M_mix = self.discriminator(self.images_I_ref_I_M_mix, reuse=True)
            """ dsc_I_ref_I_M_mix: real/fake, of shape (64, 1) """
            if self.useIRefAndMixForGanLoss:
                self.dsc_I_ref_hat = self.discriminator(self.images_I_ref_hat, reuse=True)

            # just for logging purposes:
            dsc_I_ref_sigm = tf.nn.sigmoid(self.dsc_I_ref)
            dsc_I_ref_I_M_mix_sigm = tf.nn.sigmoid(self.dsc_I_ref_I_M_mix)
            self.dsc_I_ref_mean = tf.reduce_mean(dsc_I_ref_sigm)
            self.dsc_I_ref_I_M_mix_mean = tf.reduce_mean(dsc_I_ref_I_M_mix_sigm)
            self.v_g_d = tf.reduce_mean(tf.log(dsc_I_ref_sigm) + tf.log(1 - dsc_I_ref_I_M_mix_sigm))

        with tf.variable_scope('discriminator_loss'):
            # Dsc loss x1
            self.dsc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dsc_I_ref, labels=tf.ones_like(self.dsc_I_ref)))
            print("self.dsc_loss_real: ", self.dsc_loss_real)
            # Dsc loss x3
            # this is max_D part of minmax loss function

            if self.useIRefAndMixForGanLoss:
                self.dsc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dsc_I_ref_I_M_mix, labels=tf.zeros_like(self.dsc_I_ref_I_M_mix)))
                print("self.dsc_loss_fake: ", self.dsc_loss_fake)
                self.dsc_loss_fake_Iref = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dsc_I_ref_hat, labels=tf.zeros_like(self.dsc_I_ref_hat)))
                print("self.dsc_loss_fake_Iref: ", self.dsc_loss_fake_Iref)
                self.dsc_loss = self.dsc_loss_real + (self.lambda_mix * self.dsc_loss_fake + self.lambda_ref * self.dsc_loss_fake_Iref)
            else:
                self.dsc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dsc_I_ref_I_M_mix, labels=tf.zeros_like(self.dsc_I_ref_I_M_mix)))
                print("self.dsc_loss_fake: ", self.dsc_loss_fake)
                self.dsc_loss = self.dsc_loss_real + self.dsc_loss_fake

            """ dsc_loss: a scalar, of shape () """

        with tf.variable_scope('generator_loss'):
            # D (fix Dsc you have loss for G) -> cf. Dec
            # images_x3 = Dec(f_1_2) = G(f_1_2); Dsc(images_x3) = dsc_x3
            # rationale behind g_loss: this is min_G part of minmax loss function: min log D(G(x))
            if self.useIRefAndMixForGanLoss:
                self.g_loss_mix = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dsc_I_ref_I_M_mix, labels=tf.ones_like(self.dsc_I_ref_I_M_mix)))
                self.g_loss_ref = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dsc_I_ref_hat, labels=tf.ones_like(self.dsc_I_ref_hat)))
                self.g_loss = self.lambda_mix * self.g_loss_mix + self.lambda_ref * self.g_loss_ref
            else:
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dsc_I_ref_I_M_mix, labels=tf.ones_like(self.dsc_I_ref_I_M_mix)))

        with tf.variable_scope('L1'):
            # Reconstruction loss L1 between I_ref and I_ref_hat (to ensure autoencoder works properly)
            self.rec_loss_I_ref_hat_I_ref = tf.reduce_mean(tf.abs(self.images_I_ref_hat - self.images_I_ref))
            """ rec_loss_I_ref_hat_I_ref: a scalar, of shape () """

            # Reconstruction loss L1 between t1 and t1_hat (to ensure autoencoder works properly)
            # NB: I argue that the following rec_loss is not required as every image will become I_ref eventually i.e. I_ref rec loss covers all images
            # -> just use for summary purposes
            self.rec_loss_I_t1_hat_I_t1 = tf.reduce_mean(tf.abs(self.images_t1_hat - self.images_t1))

            # L1 between I1 and I4
            self.rec_loss_I_ref_4_I_ref = tf.reduce_mean(tf.abs(self.images_I_ref_4 - self.images_I_ref))

            # L1 between t1 and t1_4
            # NB: I argue that the following rec_loss is not required as every image will become I_ref eventually i.e. I_ref rec loss covers all images
            # -> just use for summary purposes
            self.rec_loss_I_t1_4_I_t1 = tf.reduce_mean(tf.abs(self.images_t1_4 - self.images_t1))
            self.rec_loss_I_t2_4_I_t2 = tf.reduce_mean(tf.abs(self.images_t2_4 - self.images_t2))
            self.rec_loss_I_t3_4_I_t3 = tf.reduce_mean(tf.abs(self.images_t3_4 - self.images_t3))
            self.rec_loss_I_t4_4_I_t4 = tf.reduce_mean(tf.abs(self.images_t4_4 - self.images_t4))


        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()
        # Tf stuff (tell variables how to train..)
        self.dsc_vars = [var for var in t_vars if 'discriminator' in var.name and 'd_' in var.name] # discriminator
        self.gen_vars = [var for var in t_vars if 'generator' in var.name and 'g_' in var.name] # encoder + decoder (generator)
        self.cls_vars = [var for var in t_vars if 'c_' in var.name] # classifier
        self.print_model_params(t_vars)

        # save the weights
        self.saver = tf.train.Saver(self.dsc_vars + self.gen_vars + self.cls_vars + batch_norm.shadow_variables, max_to_keep=5)
        if self.params.is_train:
            self.saver_metrics = tf.train.Saver(self.dsc_vars + self.gen_vars + self.cls_vars + batch_norm.shadow_variables, max_to_keep=None)
        print("build_model() ------------------------------------------<")
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
        print('c_learning_rate: %s' % self.c_learning_rate)

        lambda_L2 = params.lambda_L2 # initial: 0.996
        lambda_Ladv = params.lambda_Ladv # initial: 0.002
        lambda_Lcls = params.lambda_Lcls # initial: 0.002
        losses_l2 = self.rec_loss_I_ref_hat_I_ref + self.rec_loss_I_ref_4_I_ref
        g_loss_comp = lambda_L2 * losses_l2 + lambda_Ladv * self.g_loss + lambda_Lcls * self.cls_loss

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

        assert not (params.continue_from and params.initialize_from), "set only one"
        load_from = None
        initialize_only = False
        if params.continue_from:
            load_from = params.continue_from
            print("continue from: %s..." % load_from)
        elif params.initialize_from:
            load_from = params.initialize_from
            initialize_only = True
            print("initialize from: %s..." % load_from)

        if load_from:
            ckpt_name = self.load(params, load_from, params.continue_from_iteration)
            iteration = int(ckpt_name[ckpt_name.rfind('-')+1:])
            if initialize_only:
                print('initializing from \'%s\'...' % ckpt_name)
                iteration = 0
            else:
                print('continuing from \'%s\'...' % ckpt_name)
            global_step.load(iteration) # load new initial value into variable

        # simple mechanism to coordinate the termination of a set of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        caccop1, caccop2, caccop3, caccop4 = self.make_summary_ops(g_loss_comp, losses_l2)

        self.initialize_uninitialized(tf.global_variables(), "global")
        self.initialize_uninitialized(tf.local_variables(), "local")

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(params.summary_dir)
        summary_writer.add_graph(self.sess.graph)

        update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS)

        try:
            signal.signal(signal.SIGTERM, self.handle_exit)

            iter_per_epoch = (self.params.num_images / self.batch_size)

            last_epoch = int(iteration // iter_per_epoch) + 1

            # Training
            while not coord.should_stop():
                # Update D and G network
                # exp69/70: do GEN update 2x
                self.sess.run([g_optim])
                self.sess.run([g_optim])

                self.sess.run([caccop1, caccop2, caccop3, caccop4, c_optim])
                self.sess.run([d_optim])

                iteration += 1

                epoch = int(iteration // iter_per_epoch) + 1
                print('iteration: %s, epoch: %d' % (str(iteration), epoch))

                if iteration % 100 == 0:
                    _,_,_,_, summary_str = self.sess.run([caccop1, caccop2, caccop3, caccop4, summary_op])
                    summary_writer.add_summary(summary_str, iteration)

                if np.mod(iteration, 500) == 1:
                    self.dump_images(iteration)

                if iteration > 1 and np.mod(iteration, 500) == 0:
                    self.save(params.checkpoint_dir, iteration)

                if epoch > last_epoch:
                    self.save_metrics(last_epoch)
                    last_epoch = epoch

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
                tb = traceback.format_exc()
                print(tb)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iteration > 0:
                self.save(params.checkpoint_dir, iteration) # save model again
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)
        # END of train()


    def test(self, params):
        """Test DCGAN"""
        """For each image in the test set create a mixed scene and save it (ie run for 1 epoch)."""

        print("test -->")

        fid_model_dir = os.path.join(params.log_dir, params.test_from, params.metric_model_folder)
        print('Loading variables from ' + fid_model_dir)
        ckpt = tf.train.get_checkpoint_state(fid_model_dir)
        if ckpt and params.metric_model_iteration:
            # Restores dump of given iteration
            ckpt_name = self.model_name + '-' + str(params.metric_model_iteration)
        else:
            raise Exception(" [!] Testing, but %s not found" % fid_model_dir)
        ckpt_file = os.path.join(fid_model_dir, ckpt_name)
        params.test_from_file = ckpt_file
        print('Reading variables to be restored from ' + ckpt_file)
        self.saver.restore(self.sess, ckpt_file)
        print('use model \'%s\'...' % ckpt_name)

        self.initialize_uninitialized(tf.global_variables(), "global")
        self.initialize_uninitialized(tf.local_variables(), "local")

        # simple mechanism to coordinate the termination of a set of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            signal.signal(signal.SIGTERM, self.handle_exit)

            num_gen_imgs = 0

            file_out_dir = params.metric_fid_out_dir
            file_all_out_dir = params.metric_fid_out_dir_all
            file_all_grid = [1, 7]
            img_all_range = randint(0, 9000)

            # for spectral normalization: initialize parameters u,v (i.e. left and right singular vectors of W)
            update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS)
            for update_op in update_ops:
                self.sess.run(update_op)

            csv_file = os.path.join(params.metric_fid_dir, 'filenames_test_%s_ep%s.csv' % (params.test_from, str(params.metric_model_iteration)))
            with open(csv_file, mode='w') as csvf:
                csv_writer = csv.writer(csvf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                iteration = 1

                while not coord.should_stop():
                    images_mix, images_mix_gen, images_Iref, images_t1, images_t2, images_t3, images_t4, ass_actual, \
                    fnames_Iref, fnames_t1, fnames_t2, fnames_t3, fnames_t4 = \
                        self.sess.run([self.images_I_M_mix, self.images_I_ref_I_M_mix, self.images_I_ref, self.images_t1, self.images_t2, self.images_t3, self.images_t4, self.assignments_actual, \
                                       self.fnames_I_ref, self.images_t1_fnames, self.images_t2_fnames, self.images_t3_fnames, self.images_t4_fnames])

                    if self.params.dump_testset_only:
                        self.dump_testset(images_Iref, images_mix, images_mix_gen, ass_actual, iteration)
                    else:
                        for i in range(self.batch_size): # for each image in batch
                            num_gen_imgs = num_gen_imgs + 1
                            img_mix = images_mix[i]
                            img_mix_gen = images_mix_gen[i]
                            fIr = d(fnames_Iref[i])
                            ft1 = d(fnames_t1[i])
                            ft2 = d(fnames_t2[i])
                            ft3 = d(fnames_t3[i])
                            ft4 = d(fnames_t4[i])

                            ass_actual_i = ass_actual[i]
                            ass_str_i = ''
                            for ass in ass_actual_i:
                                ass_str_i += str(ass)

                            # print file in folder 'images' for later metrics calculations
                            fname_mix = 'img_mix_gen_%s.png' % num_gen_imgs
                            t_name = os.path.join(file_out_dir, fname_mix)
                            imsave(t_name, img_mix_gen)

                            fname_fmix = 'img_mix_%s.png' % num_gen_imgs
                            tf_name = os.path.join(params.metric_fid_out_mixed_feature, fname_fmix)
                            imsave(tf_name, img_mix)

                            if img_all_range <= num_gen_imgs < (img_all_range + 150): # dump 150 images
                                # print all files involved in the mix into separate folder 'images_all' for showcases
                                file = "%s-%s-%s-%s-%s-%s-%s" % (fIr, ft1, ft2, ft3, ft4, ass_str_i, fname_mix)
                                out_dir = os.path.join(file_all_out_dir, file)
                                save_images_7cols(images_Iref[i], images_t1[i], images_t2[i], images_t3[i], images_t4[i], img_mix, img_mix_gen, file_all_grid, None, out_dir, addSpacing=4)
                            csv_writer.writerow([fIr, ft1, ft2, ft3, ft4, ass_str_i, fname_mix])

                            if num_gen_imgs % 300 == 0:
                                print(num_gen_imgs)

                    if self.end:
                        print('going to shutdown now...')
                        break

                    iteration += 1

        except Exception as e:
            if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Exception here, ending training..')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(e)
                tb = traceback.format_exc()
                print(tb)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

        # END of test()
        print("test <--")


    def path(self, filename):
        return os.path.join(self.params.summary_dir, filename)

    def discriminator(self, image, keep_prob=0.5, reuse=False, y=None):
        assert self.params.discriminator_coordconv is not self.params.discriminator_patchgan

        if self.params.discriminator_coordconv:
            return self.discriminator_coordconv(image, keep_prob, reuse, y)

        if self.params.discriminator_patchgan:
            return self.discriminator_patchgan(image, reuse)

        return self.discriminator_std(image, keep_prob, reuse, y)

    def discriminator_std(self, image, keep_prob=0.5, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # cf. DCGAN impl https://github.com/carpedm20/DCGAN-tensorflow.git
        h0 = lrelu(conv2d(image, self.df_dim, use_spectral_norm=True, name='d_1_h0_conv'))
        h1 = lrelu(conv2d(h0, self.df_dim*2, use_spectral_norm=True, name='d_1_h1_conv'))

        h2 = lrelu(conv2d(h1, self.df_dim * 4, use_spectral_norm=True, name='d_1_h2_conv'))

        #################################
        ch = self.df_dim*4
        x = h2
        h2 = attention(x, ch, sn=True, scope="d_attention", reuse=reuse)
        #################################

        h3 = lrelu(conv2d(h2, self.df_dim * 8, use_spectral_norm=True, name='d_1_h3_conv'))

        # NB: k=1,d=1 is like an FC layer -> to strengthen h3, to give it more capacity
        h3 = lrelu(conv2d(h3, self.df_dim*8,k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='d_1_h4_conv'))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, use_spectral_norm=True, name='d_1_h4_lin')

        # return tf.nn.sigmoid(h4)
        return h4


    def discriminator_coordconv(self, image, keep_prob=0.5, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # cf. DCGAN impl https://github.com/carpedm20/DCGAN-tensorflow.git

        h0 = lrelu(conv2d(coord_conv(image), self.df_dim, use_spectral_norm=True, name='d_1_h0_conv'))
        h1 = lrelu(conv2d(coord_conv(h0), self.df_dim*2, use_spectral_norm=True, name='d_1_h1_conv'))

        h2 = lrelu(conv2d(coord_conv(h1), self.df_dim * 4, use_spectral_norm=True, name='d_1_h2_conv'))

        #################################
        ch = self.df_dim*4
        x = h2
        h2 = attention(x, ch, sn=True, scope="d_attention", reuse=reuse)
        #################################

        h3 = lrelu(conv2d(coord_conv(h2), self.df_dim * 8, use_spectral_norm=True, name='d_1_h3_conv'))

        # NB: k=1,d=1 is like an FC layer -> to strengthen h3, to give it more capacity
        h3 = lrelu(conv2d(coord_conv(h3), self.df_dim*8,k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='d_1_h4_conv'))

        # TODO not sure if linear layer should be replaced with another CoordConv layer?
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, use_spectral_norm=True, name='d_1_h4_lin')

        # return tf.nn.sigmoid(h4)
        return h4


    def discriminator_patchgan(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        dsc = Deep_PatchGAN_Discrminator(addCoordConv=False)

        res = dsc(image)
        print('dsc: ', res.shape)

        return res


    def classifier(self, images_I_mix, images_I_ref, images_I_t1, images_I_t2, images_I_t3, images_I_t4, reuse=False):
        return self.classifier_six_image(images_I_mix, images_I_ref, images_I_t1, images_I_t2, images_I_t3, images_I_t4, reuse)


    def classifier_six_image(self, images_I_mix, images_I_ref, images_I_t1, images_I_t2, images_I_t3, images_I_t4, reuse=False):
        """From paper:
        For the classifier, we use AlexNet with batch normalization after each
        convolutional layer, but we do not use any dropout. The image inputs of
        the classifier are concatenated along the RGB channels.

        returns: a 1D matrix of size NUM_TILES i.e. (batch_size, NUM_TILES)
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        concatenated = tf.concat(axis=3, values=[images_I_mix, images_I_ref, images_I_t1, images_I_t2, images_I_t3, images_I_t4])
        assert concatenated.shape[0] == self.batch_size
        assert concatenated.shape[1] == self.image_size
        assert concatenated.shape[2] == self.image_size
        assert concatenated.shape[3] == 3 * 6

        b = True
        if b:
            assert False, "check sigmoid!"

        return self.alexnet_impl(concatenated, images_I_mix, reuse)


    def classifier_two_image(self, images_I_mix, images_I_ti, reuse=False):
        """From paper:
        For the classifier, we use AlexNet with batch normalization after each
        convolutional layer, but we do not use any dropout. The image inputs of
        the classifier are concatenated along the RGB channels.

        returns: a 1D matrix of size NUM_TILES i.e. (batch_size, NUM_TILES)
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        concatenated = tf.concat(axis=3, values=[images_I_mix, images_I_ti])
        assert concatenated.shape[0] == self.batch_size
        assert concatenated.shape[1] == self.image_size
        assert concatenated.shape[2] == self.image_size
        assert concatenated.shape[3] == 3 * 2

        return self.alexnet_impl(concatenated, images_I_mix, reuse)


    def alexnet_impl(self, concatenated, images_I_mix, reuse=False):
        conv1 = self.c_bn1(conv(concatenated, 96, 8,8,2,2, padding='VALID', name='c_3_s0_conv'))
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='c_3_mp0')

        conv2 = self.c_bn2(conv(pool1, 256, 5,5,1,1, groups=2, name='c_3_conv2')) # o: 256 1. 160
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='c_3_pool2')

        conv3 = self.c_bn3(conv(pool2, 384, 3, 3, 1, 1, name='c_3_conv3')) # o: 384 1. 288

        conv4 = self.c_bn4(conv(conv3, 384, 3, 3, 1, 1, groups=2, name='c_3_conv4')) # o: 384 1. 288

        conv5 = self.c_bn5(conv(conv4, 256, 3, 3, 1, 1, groups=2, name='c_3_conv5')) # o: 256 1. 160

        # Comment 64: because of img size 64 I had to change this max_pool here..
        # --> undo this as soon as size 128 is used again...
        assert images_I_mix.shape[1] == 64
        # pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='c_3_pool5')
        # reduces size from (32, 2, 2, 256) to (32, 1, 1, 256)
        pool5 = max_pool(conv5, 2, 2, 1, 1, padding='VALID', name='c_3_pool5')

        fc6 = tf.nn.relu(linear(tf.reshape(pool5, [self.batch_size, -1]), 4096, name='c_3_fc6') ) # o: 4096 1. 3072

        fc7 = tf.nn.relu(linear(tf.reshape(fc6, [self.batch_size, -1]), 4096, name='c_3_fc7') ) # o: 4096 1. 3072

        self.fc8 = linear(tf.reshape(fc7, [self.batch_size, -1]), NUM_TILES_L2_MIX, name='c_3_fc8')

        # return tf.nn.sigmoid(self.fc8)
        return self.fc8


    def encoder(self, tile_image, reuse=False):
        return self.encoder_conv(tile_image)


    def encoder_conv(self, tile_image, reuse=False):
        """
        returns: 1D vector f1 with size=self.feature_size
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s0 = lrelu(instance_norm(conv2d(tile_image, self.df_dim, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, self.df_dim * 2, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, self.df_dim * 4, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv2')))
        s3 = lrelu(instance_norm(conv2d(s2, self.df_dim * 8, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv3')))
        # s4 = lrelu(instance_norm(conv2d(s3, self.df_dim * 4, k_h=2, k_w=2, d_h=1, d_w=1, use_spectral_norm=True, name='g_1_conv4')))
        s4 = lrelu(instance_norm(conv2d(s3, self.df_dim * 2, k_h=2, k_w=2, d_h=1, d_w=1, use_spectral_norm=True, name='g_1_conv4')))
        # Comment 64: commented out last layer due to image size 64 (rep was too small..)
        # --> undo this as soon as size 128 is used again...
        assert tile_image.shape[1] == 32
        rep = s4
        # rep = lrelu(instance_norm(conv2d(s4, self.df_dim * 2, k_h=2, k_w=2, d_h=2, d_w=2, use_spectral_norm=True, name='g_1_conv5')))
        # TODO Qiyang: why linear layer here?
        #rep = lrelu((linear(tf.reshape(s5, [self.batch_size, -1]), self.feature_size, use_spectral_norm=True, name='g_1_fc')))

        rep = tf.reshape(rep, [self.batch_size, -1])
        assert rep.shape[0] == self.batch_size
        assert rep.shape[1] == self.feature_size_tile

        return rep


    def encoder_linear(self, tile_image, reuse=False):
        """
        returns: 1D vector f1 with size=self.feature_size
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s0 = lrelu(instance_norm(conv2d(tile_image, self.df_dim, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, self.df_dim * 2, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, self.df_dim * 4, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv2')))
        s3 = lrelu(instance_norm(conv2d(s2, self.df_dim * 6, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv3')))
        s4 = lrelu(instance_norm(conv2d(s3, self.df_dim * 8, k_h=2, k_w=2, d_h=1, d_w=1, use_spectral_norm=True, name='g_1_conv4')))

        # TODO Qiyang: why linear layer here?
        rep = lrelu((linear(tf.reshape(s4, [self.batch_size, -1]), self.feature_size_tile, use_spectral_norm=True, name='g_1_fc')))

        assert rep.shape[0] == self.batch_size
        assert rep.shape[1] == self.feature_size_tile

        return rep


    def decoder(self, inputs, preset_model, dropout_p=0.2):
        if self.params.spatial_broadcast_decoder:
            return decoder_sbd(inputs, self.image_size, self.batch_size, self.feature_size)
        else:
            return decoder_dense(inputs, self.batch_size, self.feature_size, preset_model=preset_model, dropout_p=dropout_p, apply_noise=self.useNoiseInDecoder)


    def decoder_std(self, representations, reuse=False):
        """
        returns: batch of images with size 256x60x60x3
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshape = tf.reshape(representations, [self.batch_size, 1, 1, NUM_TILES_L2_MIX * self.feature_size_tile])

        h = deconv2d(reshape, [self.batch_size, 4, 4, self.gf_dim*4], k_h=4, k_w=4, d_h=1, d_w=1, padding='VALID', use_spectral_norm=True, name='g_de_h')
        h = tf.nn.relu(h)

        h1 = deconv2d(h, [self.batch_size, 8, 8, self.gf_dim*4], use_spectral_norm=True, name='g_h1')
        h1 = tf.nn.relu(instance_norm(h1))

        h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim*4], use_spectral_norm=True, name='g_h2')
        h2 = tf.nn.relu(instance_norm(h2))

        h3 = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim*2], use_spectral_norm=True, name='g_h3')
        h3 = tf.nn.relu(instance_norm(h3))

        # #################################
        # ch = self.gf_dim*4
        # x = h3
        # h3 = attention(x, ch, sn=True, scope="g_attention", reuse=reuse)
        # #################################

        h4 = deconv2d(h3, [self.batch_size, 64, 64, self.gf_dim*1], use_spectral_norm=True, name='g_h4')
        h4 = tf.nn.relu(instance_norm(h4))

        # Comment 64: commented out last layer due to image size 64 (rep was too small..)
        # --> undo this as soon as size 128 is used again...
        assert self.image_size == 64
        #h5 = deconv2d(h4, [self.batch_size, 128, 128, self.gf_dim*1], use_spectral_norm=True, name='g_h5')
        #h5 = tf.nn.relu(instance_norm(h5))

        # h6 = deconv2d(h5, [self.batch_size, 128, 128, self.c_dim], use_spectral_norm=True, name='g_h6')
        # h6 = tf.nn.relu(instance_norm(h6))

        # From https://distill.pub/2016/deconv-checkerboard/
        # - last layer uses stride=1
        # - kernel should be divided by stride to mitigate artifacts
        #h6 = deconv2d(h5, [self.batch_size, 128, 128, self.c_dim], k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='g_h7')
        h5 = h4
        h6 = deconv2d(h5, [self.batch_size, 64, 64, self.c_dim], k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='g_h7')

        return tf.nn.tanh(h6)


    def reconstruct_I_ref_f(self, tile_id, a_tile_chunk):
        f_mix_tile_feature = self.f_I_ref_I_M_mix_hat[:, tile_id * self.feature_size_tile:(tile_id + 1) * self.feature_size_tile]
        assert f_mix_tile_feature.shape[0] == self.batch_size
        assert f_mix_tile_feature.shape[1] == self.feature_size_tile
        t_f_I_ref_tile_feature = self.I_ref_f[:, tile_id * self.feature_size_tile:(tile_id + 1) * self.feature_size_tile]
        assert t_f_I_ref_tile_feature.shape[0] == self.batch_size
        assert t_f_I_ref_tile_feature.shape[1] == self.feature_size_tile
        tile_assignments = tf.slice(self.assignments_actual, [0, tile_id], [self.batch_size, 1])
        assert tile_assignments.shape[0] == self.batch_size
        assert tile_assignments.shape[1] == 1
        f_feature_selected = tf.where(tf.equal(tile_assignments * a_tile_chunk, FROM_I_REF), f_mix_tile_feature, t_f_I_ref_tile_feature)
        assert f_feature_selected.shape == a_tile_chunk.shape
        self.I_ref_f_hat = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.I_ref_f_hat, f_feature_selected])
        return f_mix_tile_feature, tile_assignments


    def make_summary_ops(self, g_loss_comp, losses_l2):
        tf.summary.scalar('loss_g', self.g_loss)
        if self.useIRefAndMixForGanLoss:
            tf.summary.scalar('loss_g_mix', self.g_loss_mix)
            tf.summary.scalar('loss_g_ref', self.g_loss_ref)
        tf.summary.scalar('loss_g_comp', g_loss_comp)
        tf.summary.scalar('loss_L2', losses_l2)
        tf.summary.scalar('loss_cls', self.cls_loss)
        tf.summary.scalar('loss_dsc', self.dsc_loss)
        tf.summary.scalar('loss_dsc_fake', self.dsc_loss_fake)
        if self.useIRefAndMixForGanLoss:
            tf.summary.scalar('loss_dsc_fake_Iref', self.dsc_loss_fake_Iref)
        tf.summary.scalar('loss_dsc_real', self.dsc_loss_real)
        tf.summary.scalar('rec_loss_Iref_hat_I_ref', self.rec_loss_I_ref_hat_I_ref)
        tf.summary.scalar('rec_loss_I_ref_4_I_ref', self.rec_loss_I_ref_4_I_ref)
        tf.summary.scalar('rec_loss_I_t1_hat_I_t1', self.rec_loss_I_t1_hat_I_t1)
        tf.summary.scalar('rec_loss_I_t1_4_I_t1', self.rec_loss_I_t1_4_I_t1)
        tf.summary.scalar('rec_loss_I_t2_4_I_t2', self.rec_loss_I_t2_4_I_t2)
        tf.summary.scalar('rec_loss_I_t3_4_I_t3', self.rec_loss_I_t3_4_I_t3)
        tf.summary.scalar('rec_loss_I_t4_4_I_t4', self.rec_loss_I_t4_4_I_t4)
        tf.summary.scalar('psnr_images_I_ref_hat', self.images_I_ref_hat_psnr)
        tf.summary.scalar('psnr_images_I_ref_4', self.images_I_ref_4_psnr)
        tf.summary.scalar('psnr_images_t1_4', self.images_t1_4_psnr)
        tf.summary.scalar('psnr_images_t3_4', self.images_t3_4_psnr)
        tf.summary.scalar('dsc_I_ref_mean', self.dsc_I_ref_mean)
        tf.summary.scalar('dsc_I_ref_I_M_mix_mean', self.dsc_I_ref_I_M_mix_mean)
        tf.summary.scalar('V_G_D', self.v_g_d)
        tf.summary.scalar('c_learning_rate', self.c_learning_rate)

        images = tf.concat(
            tf.split(tf.concat([self.images_I_ref, self.images_I_ref_hat, self.images_I_ref_4,
                   self.images_I_M_mix, self.images_I_ref_I_M_mix], axis=2), self.batch_size,
                     axis=0), axis=1)
        tf.summary.image('images', images)

        #_ TODO add actual test images/mixes later
        #_ tf.summary.image('images_I_test_hat', self.images_I_test_hat)

        accuracy1 = tf.metrics.accuracy(predictions=tf.argmax(self.assignments_predicted_t1, 1),
                                              labels=tf.argmax(self.assignments_actual_t1, 1),
                                              updates_collections=tf.GraphKeys.UPDATE_OPS)
        tf.summary.scalar('classifier/accuracy_t1_result', accuracy1[1])
        accuracy2 = tf.metrics.accuracy(predictions=tf.argmax(self.assignments_predicted_t2, 1),
                                       labels=tf.argmax(self.assignments_actual_t2, 1),
                                       updates_collections=tf.GraphKeys.UPDATE_OPS)
        tf.summary.scalar('classifier/accuracy_t2_result', accuracy2[1])
        accuracy3 = tf.metrics.accuracy(predictions=tf.argmax(self.assignments_predicted_t3, 1),
                                       labels=tf.argmax(self.assignments_actual_t3, 1),
                                       updates_collections=tf.GraphKeys.UPDATE_OPS)
        tf.summary.scalar('classifier/accuracy_t3_result', accuracy3[1])
        accuracy4 = tf.metrics.accuracy(predictions=tf.argmax(self.assignments_predicted_t4, 1),
                                       labels=tf.argmax(self.assignments_actual_t4, 1),
                                       updates_collections=tf.GraphKeys.UPDATE_OPS)
        tf.summary.scalar('classifier/accuracy_t4_result', accuracy4[1])
        return accuracy1[1], accuracy2[1], accuracy3[1], accuracy4[1]


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.model_name)
        get_pp().pprint('[1] Save model to {} with step={}'.format(path, step))
        self.saver.save(self.sess, path, global_step=step)
        # Save model after every epoch -> is more coherent than iterations
        # if step > 1 and np.mod(step, 25000) == 0:
        #    self.save_metrics(step)

    def save_metrics(self, step):
        # save model for later FID calculation
        path = os.path.join(self.params.metric_model_dir, self.model_name)
        get_pp().pprint('[2] Save model to {} with step={}'.format(path, step))
        self.saver_metrics.save(self.sess, path, global_step=step)
        # as test calc fid directly for 5 epochs (motive: test proper persistence of all weights) -> should yield same FID!!
        # if step <= 5:
        #    print('calc FID now -->')
        #    impl....
        #    print('calc FID now <--')

    def load(self, params, load_from, iteration=None):
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(params.log_dir, load_from, params.checkpoint_folder)
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

    def dump_testset(self, images_I_ref, images_mix, images_mix_gen, ass_actual, i):
        print('dump_testset -->')

        dump_testset_dir = os.path.join(self.params.metric_fid_dir, "dump_testset")
        if not os.path.exists(dump_testset_dir):
            os.makedirs(dump_testset_dir, exist_ok=True)
            print('created dump_testset_dir: %s' % dump_testset_dir)

        st = to_string(ass_actual)
        # act_batch_size = min(self.batch_size, 1)
        act_batch_size = self.batch_size

        grid = [act_batch_size, 3]
        filename = os.path.join(dump_testset_dir, '%s_I_ref_I_mix_I_mix_gen_%s.png' % (str(i), st))
        save_images_6cols(images_I_ref, images_mix, images_mix_gen, None, None, None, grid, act_batch_size, filename, maxImg=act_batch_size, batch=True)

        print('dump_testset <--')


    def dump_images(self, counter):
        print('dump_images -->')

        img_I_ref, img_t1, img_t2, img_t3, img_t4, \
        img_I_M_mix, img_I_ref_I_M_mix, \
        img_I_ref_hat, \
        img_I_ref_4, img_t2_4, \
        ass_actual, \
        ass_actual_t1, \
        ass_actual_t2, \
        ass_actual_t3, \
        ass_actual_t4, \
        ass_pred_t1, \
        ass_pred_t2, \
        ass_pred_t3, \
        ass_pred_t4, \
        psnr_I_ref_hat, psnr_I_ref_4, psnr_t1_4, psnr_t3_4 = \
            self.sess.run([self.images_I_ref, self.images_t1, self.images_t2, self.images_t3, \
                           self.images_t4, self.images_I_M_mix, self.images_I_ref_I_M_mix, \
                           self.images_I_ref_hat, \
                           self.images_I_ref_4, self.images_t2_4, \
                           self.assignments_actual, \
                           self.assignments_actual_t1, \
                           self.assignments_actual_t2, \
                           self.assignments_actual_t3, \
                           self.assignments_actual_t4, \
                           tf.nn.sigmoid(self.assignments_predicted_t1), \
                           tf.nn.sigmoid(self.assignments_predicted_t2), \
                           tf.nn.sigmoid(self.assignments_predicted_t3), \
                           tf.nn.sigmoid(self.assignments_predicted_t4), \
                           self.images_I_ref_hat_psnr, self.images_I_ref_4_psnr, self.images_t1_4_psnr, self.images_t3_4_psnr])

        fnames_Iref, fnames_t1, fnames_t2, fnames_t3, fnames_t4 = \
            self.sess.run([self.fnames_I_ref, self.images_t1_fnames, self.images_t2_fnames, self.images_t3_fnames, self.images_t4_fnames])

        st = to_string(ass_actual)
        act_batch_size = min(self.batch_size, 16)

        grid = [act_batch_size, 5]
        save_images_5cols(img_I_ref, img_I_ref_hat, img_I_ref_4, img_I_M_mix, img_I_ref_I_M_mix, grid, act_batch_size, self.path('%s_images_I_ref_I_M_mix_%s.png' % (counter, st)), maxImg=act_batch_size)

        print("filenames iteration %d: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" % counter)
        print("filenames I_ref..: %s" % to_string2(fnames_Iref))
        print("filenames I_t1...: %s" % to_string2(fnames_t1))
        print("filenames I_t2...: %s" % to_string2(fnames_t2))
        print("filenames I_t3...: %s" % to_string2(fnames_t3))
        print("filenames I_t4...: %s" % to_string2(fnames_t4))
        print("filenames iteration %d: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" % counter)


        print('PSNR counter....: %d' % counter)
        print('PSNR I_ref_hat..: %.2f' % psnr_I_ref_hat)
        print('PSNR I_ref_4....: %.2f' % psnr_I_ref_4)
        print('PSNR I_t1_4.....: %.2f' % psnr_t1_4)
        print('PSNR I_t3_4.....: %.2f' % psnr_t3_4)

        print('assignments_actual ---------------------->>')
        print('comp.: %s' % st)
        print('t1...: %s' % to_string(ass_actual_t1))
        print('t2...: %s' % to_string(ass_actual_t2))
        print('t3...: %s' % to_string(ass_actual_t3))
        print('t4...: %s' % to_string(ass_actual_t4))
        print('assignments_actual ----------------------<<')
        print('assignments_predic ---------------------->>')
        print('t1...: %s' % to_string(ass_pred_t1))
        print('t2...: %s' % to_string(ass_pred_t2))
        print('t3...: %s' % to_string(ass_pred_t3))
        print('t4...: %s' % to_string(ass_pred_t4))
        print('assignments_predic ----------------------<<')

        print('dump_images <--')

    def print_model_params(self, t_vars):
        count_model_params(self.dsc_vars, 'Discriminator')
        g_l_exists = False
        for var in self.gen_vars:
            if 'g_1' in var.name:
                g_l_exists = True
                break
        if g_l_exists:
            enc_vars = [var for var in self.gen_vars if 'g_1' in var.name]
            dec_vars = [var for var in self.gen_vars if 'g_1' not in var.name]
            count_model_params(enc_vars, 'Generator (encoder)')
            count_model_params(dec_vars, 'Generator (decoder)')
        count_model_params(self.gen_vars, 'Generator (encoder/decoder)')
        count_model_params(self.cls_vars, 'Classifier')
        count_model_params(t_vars, 'Total')


    def initialize_uninitialized(self, vars, context):
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in vars])
        not_initialized_vars = [v for (v, f) in zip(vars, is_not_initialized) if not f]

        print("#not initialized variables '%s': %d" % (context, len(not_initialized_vars))) # only for testing
        if len(not_initialized_vars):
            print(not_initialized_vars)
            self.sess.run(tf.variables_initializer(not_initialized_vars))


def to_string2(li, elem_sep=","):
    st = ''
    for e in li:
        st += e.decode("utf-8")
        if elem_sep:
            st += elem_sep
    st = st[:-1]
    return st


def d(elem):
    return elem.decode("utf-8").split(".jpg")[0]


def to_string(ass_actual, elem_sep=None):
    st = ''
    for list in ass_actual:
        for e in list:
            st += str(e)
            if elem_sep:
                st += elem_sep
        st += '_'
    st = st[:-1]
    return st

def count_model_params(all_vars, name):
    total_parameters = 0
    for variable in all_vars:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('number of model parameters [%s]: %d' % (name, total_parameters))
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


