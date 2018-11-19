import signal
from ops_alex import *
from utils_dcgan import *
from utils_common import *
# from input_pipeline_rendered_data import get_pipeline_training_from_dump
from input_pipeline import *
from constants import *
import socket
import numpy as np
import traceback
tfd = tf.contrib.distributions



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

        self.build_model()


    def build_model(self):
        print("build_model() ------------------------------------------>")
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        tf.set_random_seed(self.random_seed)

        image_size = self.image_size

        file_train = 'datasets/coco/2017_training/tfrecords_l2mix_flip_tile_10-L2nn_4285/181115/' if 'node0' in socket.gethostname() else 'data/train-00011-of-00060.tfrecords'

        ####################################################################################
        reader = tf.TFRecordReader()
        rrm_fn = lambda name : read_record_max(name, reader)
        filenames, train_images, t1_10nn_ids, t1_10nn_subids, t1_10nn_L2, t2_10nn_ids, t2_10nn_subids, t2_10nn_L2, t3_10nn_ids, t3_10nn_subids, t3_10nn_L2, t4_10nn_ids, t4_10nn_subids, t4_10nn_L2 = \
                get_pipeline(file_train, self.batch_size, self.epochs, rrm_fn)
        print('train_images.shape..:', train_images.shape)
        self.fnames_I_ref = filenames
        self.images_I_ref = train_images

        tile_size = image_size / 2
        assert tile_size.is_integer()
        tile_size = int(tile_size)

        # create tiles for I_ref
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

        nn_id = tf.random_uniform([self.batch_size], 0, 9, dtype=tf.int32)
        path = tf.constant(self.params.tile_imgs_path)

        # t1 ############################################################################################
        path_prefix_t1 = path + tf.constant("/t1/")
        filetype = tf.constant("_t1.jpg")
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

        with tf.control_dependencies([tf.assert_equal(self.batch_size, t1_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t1_10nn_fnames), 21)]):
            t1_10nn_fnames = tf.strings.join([path_prefix_t1, t1_10nn_fnames])
            for id in range(self.batch_size):
                file = tf.read_file(t1_10nn_fnames[id])
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, tile_size, self.batch_size)
                file = tf.expand_dims(file, 0)
                t1_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t1_10nn_images, file])
        self.images_t1 = t1_10nn_images

        # t2 ############################################################################################
        path_prefix_t2 = path + tf.constant("t2/")
        filetype = tf.constant("_t2.jpg")
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

        with tf.control_dependencies([tf.assert_equal(self.batch_size, t2_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t2_10nn_fnames), 21)]):
            t2_10nn_fnames = tf.strings.join([path_prefix_t2, t2_10nn_fnames])
            for id in range(self.batch_size):
                file = tf.read_file(t2_10nn_fnames[id])
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, tile_size, self.batch_size)
                file = tf.expand_dims(file, 0)
                t2_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t2_10nn_images, file])
        self.images_t2 = t2_10nn_images

        # t3 ############################################################################################
        path_prefix_t3 = path + tf.constant("t3/")
        filetype = tf.constant("_t3.jpg")
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

        with tf.control_dependencies([tf.assert_equal(self.batch_size, t3_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t3_10nn_fnames), 21)]):
            t3_10nn_fnames = tf.strings.join([path_prefix_t3, t3_10nn_fnames])
            for id in range(self.batch_size):
                file = tf.read_file(t3_10nn_fnames[id])
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, tile_size, self.batch_size)
                file = tf.expand_dims(file, 0)
                t3_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t3_10nn_images, file])
        self.images_t3 = t3_10nn_images

        # t4 ############################################################################################
        path_prefix_t4 = path + tf.constant("t4/")
        filetype = tf.constant("_t4.jpg")
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

        with tf.control_dependencies([tf.assert_equal(self.batch_size, t4_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t4_10nn_fnames), 21)]):
            t4_10nn_fnames = tf.strings.join([path_prefix_t4, t4_10nn_fnames])
            for id in range(self.batch_size):
                file = tf.read_file(t4_10nn_fnames[id])
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, tile_size, self.batch_size)
                file = tf.expand_dims(file, 0)
                t4_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t4_10nn_images, file])
        self.images_t4 = t4_10nn_images

        # ###########################################################################################################
        # ###########################################################################################################



















        # 12.11: currently leave scaling idea out and first focus on the core clustering idea
        # rrs_def_fn = lambda name, scale : read_record_scale(name, reader, scale)
        # rrs_fn = lambda name : rrs_def_fn(name, 9) # 90% scale
        # _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        # self.images_i2 = train_images
        # rrs_fn = lambda name : rrs_def_fn(name, 8) # 80% scale
        # _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        # self.images_i3 = train_images
        # rrs_fn = lambda name : rrs_def_fn(name, 7) # 70% scale
        # _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        # self.images_i4 = train_images
        # rrs_fn = lambda name : rrs_def_fn(name, 6) # 60% scale
        # _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        # self.images_i5 = train_images
        # rrs_fn = lambda name : rrs_def_fn(name, 5) # 50% scale
        # _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        # self.images_i6 = train_images
        # rrs_fn = lambda name : rrs_def_fn(name, 4) # 40% scale
        # _, _, _, train_images = get_pipeline(file_train, self.batch_size, self.epochs, rrs_fn)
        # self.images_i7 = train_images



        ####################################################################################
        # self.images_x1 = train_images[0:self.batch_size, :, :, :]
        # """ images_x1: tensor of images (batch_size, image_size, image_size, 3) """
        # self.images_x2 = train_images[self.batch_size:self.batch_size * 2, :, :, :]

        # # image overlap arithmetic
        # overlap = 0 # self.params.slice_overlap
        # # assert overlap, 'hyperparameter \'overlap\' is not an integer'
        # slice_size = (image_size + 2 * overlap) / 2
        # assert slice_size.is_integer(), 'hyperparameter \'overlap\' invalid: %d' % overlap
        # slice_size = int(slice_size)
        # slice_size_overlap = slice_size - overlap
        # slice_size_overlap = int(slice_size_overlap)
        # print('overlap: %d, slice_size: %d, slice_size_overlap: %d' % \
        #       (overlap, slice_size, slice_size_overlap))
        tile_size = image_size / 2
        assert tile_size.is_integer()
        tile_size = int(tile_size)


        # # create 1st tile for rest of images
        # self.i2_tile1 = tf.image.crop_to_bounding_box(self.images_i2, 0, 0, slice_size, slice_size)
        # self.i3_tile1 = tf.image.crop_to_bounding_box(self.images_i3, 0, 0, slice_size, slice_size)
        # self.i4_tile1 = tf.image.crop_to_bounding_box(self.images_i4, 0, 0, slice_size, slice_size)
        # self.i5_tile1 = tf.image.crop_to_bounding_box(self.images_i5, 0, 0, slice_size, slice_size)
        # self.i6_tile1 = tf.image.crop_to_bounding_box(self.images_i6, 0, 0, slice_size, slice_size)
        # self.i7_tile1 = tf.image.crop_to_bounding_box(self.images_i7, 0, 0, slice_size, slice_size)
        #
        # # create 2nd tile for rest of images
        # self.i2_tile2 = tf.image.crop_to_bounding_box(self.images_i2, 0, slice_size_overlap, slice_size, slice_size)
        # self.i3_tile2 = tf.image.crop_to_bounding_box(self.images_i3, 0, slice_size_overlap, slice_size, slice_size)
        # self.i4_tile2 = tf.image.crop_to_bounding_box(self.images_i4, 0, slice_size_overlap, slice_size, slice_size)
        # self.i5_tile2 = tf.image.crop_to_bounding_box(self.images_i5, 0, slice_size_overlap, slice_size, slice_size)
        # self.i6_tile2 = tf.image.crop_to_bounding_box(self.images_i6, 0, slice_size_overlap, slice_size, slice_size)
        # self.i7_tile2 = tf.image.crop_to_bounding_box(self.images_i7, 0, slice_size_overlap, slice_size, slice_size)
        #
        # # create 3rd tile for rest of images
        # self.i2_tile3 = tf.image.crop_to_bounding_box(self.images_i2, slice_size_overlap, 0, slice_size, slice_size)
        # self.i3_tile3 = tf.image.crop_to_bounding_box(self.images_i3, slice_size_overlap, 0, slice_size, slice_size)
        # self.i4_tile3 = tf.image.crop_to_bounding_box(self.images_i4, slice_size_overlap, 0, slice_size, slice_size)
        # self.i5_tile3 = tf.image.crop_to_bounding_box(self.images_i5, slice_size_overlap, 0, slice_size, slice_size)
        # self.i6_tile3 = tf.image.crop_to_bounding_box(self.images_i6, slice_size_overlap, 0, slice_size, slice_size)
        # self.i7_tile3 = tf.image.crop_to_bounding_box(self.images_i7, slice_size_overlap, 0, slice_size, slice_size)
        #
        # # create 4th tile for rest of images
        # self.i2_tile4 = tf.image.crop_to_bounding_box(self.images_i2, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        # self.i3_tile4 = tf.image.crop_to_bounding_box(self.images_i3, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        # self.i4_tile4 = tf.image.crop_to_bounding_box(self.images_i4, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        # self.i5_tile4 = tf.image.crop_to_bounding_box(self.images_i5, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        # self.i6_tile4 = tf.image.crop_to_bounding_box(self.images_i6, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
        # self.i7_tile4 = tf.image.crop_to_bounding_box(self.images_i7, slice_size_overlap, slice_size_overlap, slice_size, slice_size)



        self.chunk_num = self.params.chunk_num
        """ number of chunks: 8 """
        self.chunk_size = self.params.chunk_size
        """ size per chunk: 64 """
        self.feature_size = self.chunk_size*self.chunk_num
        """ equals the size of all chunks from a single tile """

        # each tile chunk is initialized with 1's (batch_size,256)
        a_tile_chunk = tf.ones((self.batch_size, self.feature_size), dtype=tf.int32)
        assert a_tile_chunk.shape[0] == self.batch_size
        assert a_tile_chunk.shape[1] == self.feature_size

        with tf.variable_scope('generator') as scope_generator:
            self.I_ref_f1 = self.encoder(self.I_ref_t1)

            self.f_I_ref_composite = tf.zeros((self.batch_size, NUM_TILES_L2_MIX * self.feature_size))
            # this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.decoder(self.f_I_ref_composite)

            # Classifier
            # -> this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.classifier(self.I_ref_t1, self.I_ref_t1, self.I_ref_t1, self.I_ref_t1
                            , self.I_ref_t1, self.I_ref_t1, self.I_ref_t1, self.I_ref_t1
                            , self.I_ref_t1, self.I_ref_t1, self.I_ref_t1, self.I_ref_t1)

            # to share the weights between the Encoders
            scope_generator.reuse_variables()

            self.I_ref_f2 = self.encoder(self.I_ref_t2)
            self.I_ref_f3 = self.encoder(self.I_ref_t3)
            self.I_ref_f4 = self.encoder(self.I_ref_t4)

            self.t1_f = self.encoder(self.images_t1)
            self.t2_f = self.encoder(self.images_t2)
            self.t3_f = self.encoder(self.images_t3)
            self.t4_f = self.encoder(self.images_t4)

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
                tile_1 = tf.expand_dims(tf.where(cond_Iref_t1, self.I_ref_t1[id], self.images_t1[id]), 0)
                # for the assignment mask e.g. [0 1 1 0], of shape (4,)
                # 0 selects the corresponding tile from I_ref
                # 1 selects the corresponding tile from I_M
                assignment_1 = tf.where(cond_Iref_t1, 0, 1)
                self.J_1_tile = tile_1 if id == 0 else tf.concat(axis=0, values=[self.J_1_tile, tile_1])
                feature_1 = tf.expand_dims(tf.where(cond_Iref_t1, self.I_ref_f1[id], self.t1_f[id]), 0)
                # self.J_1_f = feature_1 if id == 0 else tf.concat(axis=0, values=[self.J_1_f, feature_1])

                is_t2_maxL2 = tf.equal(argmax_L2, 1)
                is_t2_minL2 = tf.equal(argmin_L2, 1)
                cond_Iref_t2 = tf.logical_and(tf.logical_or(is_t2_maxL2, tf.greater(t2_10nn_L2_b, tau)), tf.logical_not(is_t2_minL2))
                tile_2 = tf.expand_dims(tf.where(cond_Iref_t2, self.I_ref_t2[id], self.images_t2[id]), 0)
                assignment_2 = tf.where(cond_Iref_t2, 0, 1)
                self.J_2_tile = tile_2 if id == 0 else tf.concat(axis=0, values=[self.J_2_tile, tile_2])
                feature_2 = tf.expand_dims(tf.where(cond_Iref_t2, self.I_ref_f2[id], self.t2_f[id]), 0)
                # self.J_2_f = feature_2 if id == 0 else tf.concat(axis=0, values=[self.J_2_f, feature_2])

                is_t3_maxL2 = tf.equal(argmax_L2, 2)
                is_t3_minL2 = tf.equal(argmin_L2, 2)
                cond_Iref_t3 = tf.logical_and(tf.logical_or(is_t3_maxL2, tf.greater(t3_10nn_L2_b, tau)), tf.logical_not(is_t3_minL2))
                tile_3 = tf.expand_dims(tf.where(cond_Iref_t3, self.I_ref_t3[id], self.images_t3[id]), 0)
                assignment_3 = tf.where(cond_Iref_t3, 0, 1)
                self.J_3_tile = tile_3 if id == 0 else tf.concat(axis=0, values=[self.J_3_tile, tile_3])
                feature_3 = tf.expand_dims(tf.where(cond_Iref_t3, self.I_ref_f3[id], self.t3_f[id]), 0)
                # self.J_3_f = feature_3 if id == 0 else tf.concat(axis=0, values=[self.J_3_f, feature_3])

                is_t4_maxL2 = tf.equal(argmax_L2, 3)
                is_t4_minL2 = tf.equal(argmin_L2, 3)
                cond_Iref_t4 = tf.logical_and(tf.logical_or(is_t4_maxL2, tf.greater(t4_10nn_L2_b, tau)), tf.logical_not(is_t4_minL2))
                tile_4 = tf.expand_dims(tf.where(cond_Iref_t4, self.I_ref_t4[id], self.images_t4[id]), 0)
                assignment_4 = tf.where(cond_Iref_t4, 0, 1)
                self.J_4_tile = tile_4 if id == 0 else tf.concat(axis=0, values=[self.J_4_tile, tile_4])
                feature_4 = tf.expand_dims(tf.where(cond_Iref_t4, self.I_ref_f4[id], self.t4_f[id]), 0)
                # self.J_4_f = feature_4 if id == 0 else tf.concat(axis=0, values=[self.J_4_f, feature_4])

                assignments = tf.stack(axis=0, values=[assignment_1, assignment_2, assignment_3, assignment_4])
                assignments = tf.expand_dims(tf.reshape(assignments, [-1]), 0)
                self.assignments_actual = assignments if id == 0 else tf.concat(axis=0, values=[self.assignments_actual, assignments])  # or 'mask'

                assert feature_1.shape[0] == 1
                assert feature_1.shape[1] == self.feature_size
                assert feature_1.shape[0] == feature_2.shape[0] and feature_1.shape[1] == feature_2.shape[1]
                assert feature_2.shape[0] == feature_3.shape[0] and feature_2.shape[1] == feature_3.shape[1]
                assert feature_2.shape[0] == feature_4.shape[0] and feature_2.shape[1] == feature_4.shape[1]
                assert feature_1.shape[1] == a_tile_chunk.shape[1]
                f_features_selected = tf.concat(axis=0, values=[feature_1, feature_2, feature_3, feature_4])  # axis=1
                f_features_selected = tf.reshape(f_features_selected, [-1])
                f_features_selected = tf.expand_dims(f_features_selected, 0)
                self.f_I_ref_I_M_mix = f_features_selected if id == 0 else tf.concat(axis=0, values=[self.f_I_ref_I_M_mix, f_features_selected])

            assert self.J_1_tile.shape[0] == self.batch_size
            assert self.J_1_tile.shape[1] == tile_size
            assert self.J_1_tile.shape[2] == tile_size
            assert self.J_1_tile.shape[3] == 3
            assert self.J_1_tile.shape == self.J_2_tile.shape
            assert self.J_2_tile.shape == self.J_3_tile.shape
            assert self.J_2_tile.shape == self.J_4_tile.shape
            assert self.J_1_tile.shape == self.images_t1.shape
            assert self.assignments_actual.shape[0] == self.batch_size
            assert self.assignments_actual.shape[1] == NUM_TILES_L2_MIX
            assert self.f_I_ref_I_M_mix.shape[0] == self.batch_size
            assert self.f_I_ref_I_M_mix.shape[1] == self.feature_size * NUM_TILES_L2_MIX





            # # 1. Determine L2 closest features wrt reference image I1
            # # choose feature (tile) with miminum L2 distance to feature of I1 tile1
            # with tf.variable_scope('L2_tile1_selection'):
            #     all_t1 = tf.concat([self.i2_tile1, self.i3_tile1, self.i4_tile1, self.i5_tile1, self.i6_tile1, self.i7_tile1], axis=0)
            #     all_f1 = tf.concat([self.i2_f_1, self.i3_f_1, self.i4_f_1, self.i5_f_1, self.i6_f_1, self.i7_f_1], axis=0)
            #     for i in range(self.batch_size):
            #         I1_f_1_i = self.I_ref_f1[i]
            #         rec_loss_f1_f2 = tf.reduce_mean(tf.square(I1_f_1_i - self.i2_f_1), 1)
            #         rec_loss_f1_f3 = tf.reduce_mean(tf.square(I1_f_1_i - self.i3_f_1), 1)
            #         rec_loss_f1_f4 = tf.reduce_mean(tf.square(I1_f_1_i - self.i4_f_1), 1)
            #         rec_loss_f1_f5 = tf.reduce_mean(tf.square(I1_f_1_i - self.i5_f_1), 1)
            #         rec_loss_f1_f6 = tf.reduce_mean(tf.square(I1_f_1_i - self.i6_f_1), 1)
            #         rec_loss_f1_f7 = tf.reduce_mean(tf.square(I1_f_1_i - self.i7_f_1), 1)
            #
            #         all = tf.stack(axis=0, values=[rec_loss_f1_f2, rec_loss_f1_f3, rec_loss_f1_f4, rec_loss_f1_f5, rec_loss_f1_f6, rec_loss_f1_f7])
            #         all = tf.reshape(all, [-1])
            #         # argmin_i holds the index of the L2 closest feature (tile) wrt feature I1_f_1_i across all pipelines within the same batch
            #         argmin_i = tf.argmin(all, axis=0)
            #         tile_1_i = all_t1[argmin_i]
            #         tile_1_i = tf.expand_dims(tile_1_i, 0)
            #         self.J_1_tile = tile_1_i if i == 0 else tf.concat(axis=0, values=[self.J_1_tile, tile_1_i])
            #         f_1_i = all_f1[argmin_i]
            #         f_1_i = tf.expand_dims(f_1_i, 0)
            #         self.J_1_f = f_1_i if i == 0 else tf.concat(axis=0, values=[self.J_1_f, f_1_i])
            #
            #     assert self.J_1_tile.shape[0] == self.batch_size
            #     assert self.J_1_tile.shape[1] == int(self.image_size / 2)
            #     assert self.J_1_tile.shape[2] == int(self.image_size / 2)
            #     assert self.J_1_tile.shape[3] == self.c_dim
            #     assert self.J_1_f.shape[0] == self.batch_size
            #     assert self.J_1_f.shape[1] == self.feature_size
            #
            #
            # with tf.variable_scope('L2_tile2_selection'):
            #     all_t2 = tf.concat([self.i2_tile2, self.i3_tile2, self.i4_tile2, self.i5_tile2, self.i6_tile2, self.i7_tile2], axis=0)
            #     all_f2 = tf.concat([self.i2_f_2, self.i3_f_2, self.i4_f_2, self.i5_f_2, self.i6_f_2, self.i7_f_2], axis=0)
            #     for i in range(self.batch_size):
            #         I1_f_2_i = self.I_ref_f2[i]
            #         rec_loss_f2_f2 = tf.reduce_mean(tf.square(I1_f_2_i - self.i2_f_2), 1)
            #         rec_loss_f2_f3 = tf.reduce_mean(tf.square(I1_f_2_i - self.i3_f_2), 1)
            #         rec_loss_f2_f4 = tf.reduce_mean(tf.square(I1_f_2_i - self.i4_f_2), 1)
            #         rec_loss_f2_f5 = tf.reduce_mean(tf.square(I1_f_2_i - self.i5_f_2), 1)
            #         rec_loss_f2_f6 = tf.reduce_mean(tf.square(I1_f_2_i - self.i6_f_2), 1)
            #         rec_loss_f2_f7 = tf.reduce_mean(tf.square(I1_f_2_i - self.i7_f_2), 1)
            #
            #         all = tf.stack(axis=0, values=[rec_loss_f2_f2, rec_loss_f2_f3, rec_loss_f2_f4, rec_loss_f2_f5, rec_loss_f2_f6, rec_loss_f2_f7])
            #         all = tf.reshape(all, [-1])
            #         # argmin_i holds the index of the L2 closest feature wrt feature I1_f_2_i across all pipelines within the same batch
            #         argmin_i = tf.argmin(all, axis=0)
            #         tile_2_i = all_t2[argmin_i]
            #         tile_2_i = tf.expand_dims(tile_2_i, 0)
            #         self.J_2_tile = tile_2_i if i == 0 else tf.concat(axis=0, values=[self.J_2_tile, tile_2_i])
            #         f_i = all_f2[argmin_i]
            #         f_i = tf.expand_dims(f_i, 0)
            #         self.J_2_f = f_i if i == 0 else tf.concat(axis=0, values=[self.J_2_f, f_i])
            #
            #     assert self.J_2_tile.shape[0] == self.batch_size
            #     assert self.J_2_tile.shape[1] == int(self.image_size / 2)
            #     assert self.J_2_tile.shape[2] == int(self.image_size / 2)
            #     assert self.J_2_tile.shape[3] == self.c_dim
            #     assert self.J_2_f.shape[0] == self.batch_size
            #     assert self.J_2_f.shape[1] == self.feature_size
            #
            #
            # with tf.variable_scope('L2_tile3_selection'):
            #     all_t3 = tf.concat([self.i2_tile3, self.i3_tile3, self.i4_tile3, self.i5_tile3, self.i6_tile3, self.i7_tile3], axis=0)
            #     all_f3 = tf.concat([self.i2_f_3, self.i3_f_3, self.i4_f_3, self.i5_f_3, self.i6_f_3, self.i7_f_3], axis=0)
            #     for i in range(self.batch_size):
            #         I1_f_3_i = self.I_ref_f3[i]
            #         rec_loss_f3_f2 = tf.reduce_mean(tf.square(I1_f_3_i - self.i2_f_3), 1)
            #         rec_loss_f3_f3 = tf.reduce_mean(tf.square(I1_f_3_i - self.i3_f_3), 1)
            #         rec_loss_f3_f4 = tf.reduce_mean(tf.square(I1_f_3_i - self.i4_f_3), 1)
            #         rec_loss_f3_f5 = tf.reduce_mean(tf.square(I1_f_3_i - self.i5_f_3), 1)
            #         rec_loss_f3_f6 = tf.reduce_mean(tf.square(I1_f_3_i - self.i6_f_3), 1)
            #         rec_loss_f3_f7 = tf.reduce_mean(tf.square(I1_f_3_i - self.i7_f_3), 1)
            #
            #         all = tf.stack(axis=0, values=[rec_loss_f3_f2, rec_loss_f3_f3, rec_loss_f3_f4, rec_loss_f3_f5, rec_loss_f3_f6, rec_loss_f3_f7])
            #         all = tf.reshape(all, [-1])
            #         # argmin_i holds the index of the L2 closest feature wrt feature I1_f_3_i across all pipelines within the same batch
            #         argmin_i = tf.argmin(all, axis=0)
            #         tile_3_i = all_t3[argmin_i]
            #         tile_3_i = tf.expand_dims(tile_3_i, 0)
            #         self.J_3_tile = tile_3_i if i == 0 else tf.concat(axis=0, values=[self.J_3_tile, tile_3_i])
            #         f_i = all_f3[argmin_i]
            #         f_i = tf.expand_dims(f_i, 0)
            #         self.J_3_f = f_i if i == 0 else tf.concat(axis=0, values=[self.J_3_f, f_i])
            #
            #     assert self.J_3_tile.shape[0] == self.batch_size
            #     assert self.J_3_tile.shape[1] == int(self.image_size / 2)
            #     assert self.J_3_tile.shape[2] == int(self.image_size / 2)
            #     assert self.J_3_tile.shape[3] == self.c_dim
            #     assert self.J_3_f.shape[0] == self.batch_size
            #     assert self.J_3_f.shape[1] == self.feature_size
            #
            #
            # with tf.variable_scope('L2_tile4_selection'):
            #     all_t4 = tf.concat([self.i2_tile4, self.i3_tile4, self.i4_tile4, self.i5_tile4, self.i6_tile4, self.i7_tile4], axis=0)
            #     all_f4 = tf.concat([self.i2_f_4, self.i3_f_4, self.i4_f_4, self.i5_f_4, self.i6_f_4, self.i7_f_4], axis=0)
            #     for i in range(self.batch_size):
            #         I1_f_4_i = self.I_ref_f4[i]
            #         rec_loss_f4_f2 = tf.reduce_mean(tf.square(I1_f_4_i - self.i2_f_4), 1)
            #         rec_loss_f4_f3 = tf.reduce_mean(tf.square(I1_f_4_i - self.i3_f_4), 1)
            #         rec_loss_f4_f4 = tf.reduce_mean(tf.square(I1_f_4_i - self.i4_f_4), 1)
            #         rec_loss_f4_f5 = tf.reduce_mean(tf.square(I1_f_4_i - self.i5_f_4), 1)
            #         rec_loss_f4_f6 = tf.reduce_mean(tf.square(I1_f_4_i - self.i6_f_4), 1)
            #         rec_loss_f4_f7 = tf.reduce_mean(tf.square(I1_f_4_i - self.i7_f_4), 1)
            #
            #         all = tf.stack(axis=0, values=[rec_loss_f4_f2, rec_loss_f4_f3, rec_loss_f4_f4, rec_loss_f4_f5, rec_loss_f4_f6, rec_loss_f4_f7])
            #         all = tf.reshape(all, [-1])
            #         # argmin_i holds the index of the L2 closest feature wrt feature I1_f_4_i across all pipelines within the same batch
            #         argmin_i = tf.argmin(all, axis=0)
            #         tile_i = all_t4[argmin_i]
            #         tile_i = tf.expand_dims(tile_i, 0)
            #         self.J_4_tile = tile_i if i == 0 else tf.concat(axis=0, values=[self.J_4_tile, tile_i])
            #         f_i = all_f4[argmin_i]
            #         f_i = tf.expand_dims(f_i, 0)
            #         self.J_4_f = f_i if i == 0 else tf.concat(axis=0, values=[self.J_4_f, f_i])
            #
            #     assert self.J_4_tile.shape[0] == self.batch_size
            #     assert self.J_4_tile.shape[1] == int(self.image_size / 2)
            #     assert self.J_4_tile.shape[2] == int(self.image_size / 2)
            #     assert self.J_4_tile.shape[3] == self.c_dim
            #     assert self.J_4_f.shape[0] == self.batch_size
            #     assert self.J_4_f.shape[1] == self.feature_size

            # ##################################################################################################################################


            # having all 4 selected tiles t_i, assemble the equivalent of images_I_M analogous to images_I_ref
            row1 = tf.concat([self.images_t1, self.images_t3], axis=1)
            row2 = tf.concat([self.images_t2, self.images_t4], axis=1)
            self.images_I_M = tf.concat([row1, row2], axis=2)
            assert self.images_I_M.shape[1] == self.images_I_ref.shape[1]
            assert self.images_I_M.shape[2] == self.images_I_ref.shape[2]
            assert self.images_I_M.shape[3] == self.images_I_ref.shape[3]

            # just for logging purposes __start ###
            row1 = tf.concat([self.J_1_tile, self.J_2_tile], axis=1)
            row2 = tf.concat([self.J_3_tile, self.J_4_tile], axis=1)
            self.images_J = tf.concat([row1, row2], axis=2)
            # just for logging purposes __end ###

            # build composite feature including all I_ref tile features
            self.f_I_ref_composite = tf.concat([self.I_ref_f1, self.I_ref_f2, self.I_ref_f3, self.I_ref_f4], 1)
            self.images_I_ref_hat = self.decoder(self.f_I_ref_composite)
            assert self.images_I_ref_hat.shape[1] == self.image_size
            # Enc/Dec for I_ref __end ##########################################

            # Enc/Dec for I_M __start ##########################################
            # build composite feature including all I_M tile features

            # self.f_I_M_composite = tf.concat([self.J_1_f, self.J_2_f, self.J_3_f, self.J_4_f], 1)
            # self.images_I_M_hat = self.decoder(self.f_I_M_composite)
            self.f_I_M_composite = tf.concat([self.t1_f, self.t2_f, self.t3_f, self.t4_f], 1)
            assert self.f_I_M_composite.shape == self.f_I_ref_composite.shape
            self.images_I_M_hat = self.decoder(self.f_I_M_composite)
            assert self.images_I_M_hat.shape == self.images_I_ref.shape
            # Enc/Dec for I_M __end ##########################################


            # # Mask handling __start ##########################################
            # # for the mask e.g. [0 1 1 0], of shape (4,)
            # # 1 selects the corresponding tile from I1
            # # 0 selects the corresponding tile from I2
            # self.mask = tfd.Bernoulli(self.params.mask_bias_x1).sample(NUM_TILES_L2_MIX)
            #
            # # each tile chunk is initialized with 1's (64,256)
            # a_tile_chunk = tf.ones((self.batch_size,self.feature_size),dtype=tf.int32)
            # assert a_tile_chunk.shape[0] == self.batch_size
            # assert a_tile_chunk.shape[1] == self.feature_size
            #
            # # mix the tile features according to the mask m
            # # for each tile slot in f_1_2 fill it from either x1 or x2
            # # tile_feature = includes all chunks from the same tile
            # for tile_id in range(0, NUM_TILES_L2_MIX): # for each tile feature slot
            #     t_f_I_ref_tile_feature = self.f_I_ref_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
            #     assert t_f_I_ref_tile_feature.shape[0] == a_tile_chunk.shape[0]
            #     assert t_f_I_ref_tile_feature.shape[1] == self.feature_size
            #     t_f_I_M_tile_feature = self.f_I_M_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
            #     assert t_f_I_M_tile_feature.shape[1] == self.feature_size
            #     assert t_f_I_M_tile_feature.shape[1] == a_tile_chunk.shape[1]
            #     tile_mask_batchsize = tf.equal(self.mask[tile_id] * a_tile_chunk, FROM_I1)
            #     assert tile_mask_batchsize.shape[0] == self.batch_size
            #     assert tile_mask_batchsize.shape[1] == self.feature_size
            #     assert tile_mask_batchsize.shape == t_f_I_ref_tile_feature.shape
            #     assert tile_mask_batchsize.shape[1] == t_f_I_M_tile_feature.shape[1]
            #     f_feature_selected = tf.where(tile_mask_batchsize, t_f_I_ref_tile_feature, t_f_I_M_tile_feature)
            #     self.f_Iref_I2_mix = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_Iref_I2_mix, f_feature_selected])
            #
            # assert self.f_Iref_I2_mix.shape[0] == self.batch_size
            # assert self.f_Iref_I2_mix.shape[1] == self.feature_size * NUM_TILES_L2_MIX
            #
            # # Mask handling __end ##########################################


            # Dec I_ref_I_M_mix
            self.images_I_ref_I_M_mix = self.decoder(self.f_I_ref_I_M_mix)

            # TODO think through this: would it be better to not crop here and just use the whole image ????

            # create tiles for I_ref_I_M_mix
            self.I_ref_I_M_tile1 = tf.image.crop_to_bounding_box(self.images_I_ref_I_M_mix, 0, 0, tile_size, tile_size)
            self.I_ref_I_M_tile2 = tf.image.crop_to_bounding_box(self.images_I_ref_I_M_mix, 0, tile_size, tile_size, tile_size)
            self.I_ref_I_M_tile3 = tf.image.crop_to_bounding_box(self.images_I_ref_I_M_mix, tile_size, 0, tile_size, tile_size)
            self.I_ref_I_M_tile4 = tf.image.crop_to_bounding_box(self.images_I_ref_I_M_mix, tile_size, tile_size, tile_size, tile_size)

            # CLS
            self.assignments_predicted = self.classifier(self.I_ref_I_M_tile1, self.I_ref_I_M_tile2, self.I_ref_I_M_tile3, self.I_ref_I_M_tile4,
                                                         self.I_ref_t1, self.I_ref_t2, self.I_ref_t3, self.I_ref_t4,
                                                         self.images_t1, self.images_t2, self.images_t3, self.images_t4)
            """ assignments_predicted is of size (batch_size, 4) """
            assert self.assignments_predicted.shape[0] == self.batch_size
            assert self.assignments_predicted.shape[1] == NUM_TILES_L2_MIX

            # # cf original mask
            # self.mask_actual = tf.cast(tf.ones((self.batch_size, NUM_TILES_L2_MIX), dtype=tf.int32) * self.mask, tf.float32)
            # """ mask_actual: mask (4,) scaled to batch_size, of shape (64, 4) """

            assert self.assignments_predicted.shape == self.assignments_actual.shape

            # f3 (Enc for f3)
            self.I_ref_I_M_f_1 = self.encoder(self.I_ref_I_M_tile1)
            self.I_ref_I_M_f_2 = self.encoder(self.I_ref_I_M_tile2)
            self.I_ref_I_M_f_3 = self.encoder(self.I_ref_I_M_tile3)
            self.I_ref_I_M_f_4 = self.encoder(self.I_ref_I_M_tile4)
            assert self.I_ref_I_M_f_4.shape == self.I_ref_I_M_f_1.shape

            # build composite feature including all I1 tile features
            self.f_I_ref_I_M_mix_hat = tf.concat([self.I_ref_I_M_f_1, self.I_ref_I_M_f_2, self.I_ref_I_M_f_3, self.I_ref_I_M_f_4], 1)
            assert self.f_I_ref_I_M_mix_hat.shape == self.f_I_ref_I_M_mix.shape
            assert self.f_I_ref_I_M_mix_hat.shape[1] == self.feature_size * NUM_TILES_L2_MIX


            # RECONSTRUCT f_I_ref_composite_hat/f_I_M_composite_hat FROM f_I_ref_I_M_mix_hat START
            for tile_id in range(0, NUM_TILES_L2_MIX):
                f_mix_tile_feature = self.f_I_ref_I_M_mix_hat[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                assert f_mix_tile_feature.shape[0] == self.batch_size
                assert f_mix_tile_feature.shape[1] == self.feature_size
                t_f_I_ref_tile_feature = self.f_I_ref_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                assert t_f_I_ref_tile_feature.shape[0] == self.batch_size
                assert t_f_I_ref_tile_feature.shape[1] == self.feature_size
                t_f_I_M_tile_feature = self.f_I_M_composite[:, tile_id * self.feature_size:(tile_id + 1) * self.feature_size]
                assert t_f_I_M_tile_feature.shape[0] == self.batch_size
                assert t_f_I_M_tile_feature.shape[1] == self.feature_size
                # TODO: perhaps double check this logic (with slice) at some later stage again or rewrite this entire for loop!
                tile_assignments = tf.slice(self.assignments_actual, [0, tile_id], [self.batch_size, 1])
                assert tile_assignments.shape[0] == self.batch_size
                assert tile_assignments.shape[1] == 1
                f_feature_selected = tf.where(tf.equal(tile_assignments * a_tile_chunk, FROM_I_REF), f_mix_tile_feature, t_f_I_ref_tile_feature)
                assert f_feature_selected.shape == a_tile_chunk.shape
                self.f_I_ref_composite_hat = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_I_ref_composite_hat, f_feature_selected])
                f_feature_selected = tf.where(tf.equal(tile_assignments * a_tile_chunk, FROM_I_M), f_mix_tile_feature, t_f_I_M_tile_feature)
                assert f_feature_selected.shape == a_tile_chunk.shape
                self.f_I_M_composite_hat = f_feature_selected if tile_id == 0 else tf.concat(axis=1, values=[self.f_I_M_composite_hat, f_feature_selected])
            assert self.f_I_ref_composite_hat.shape[0] == self.batch_size
            assert self.f_I_ref_composite_hat.shape[1] == self.feature_size * NUM_TILES_L2_MIX
            assert self.f_I_ref_composite_hat.shape == self.f_I_ref_composite.shape
            assert self.f_I_M_composite_hat.shape == self.f_I_M_composite.shape
            # RECONSTRUCT f_I_ref_composite_hat/f_I_M_composite_hat FROM f_I_ref_I_M_mix_hat END

            # decode to I_ref_4 for L2 with I_ref
            self.images_I_ref_4 = self.decoder(self.f_I_ref_composite_hat)
            """ images_I4: batch of reconstructed images I4 with shape (batch_size, 128, 128, 3) """
            # decode to I5 for L2 with I2
            self.images_I_M_5 = self.decoder(self.f_I_M_composite_hat)


        with tf.variable_scope('classifier_loss'):
            # Cls loss; mask_batchsize here is GT, cls should predict correct mask..
            self.cls_loss = binary_cross_entropy_with_logits(tf.cast(self.assignments_actual, tf.float32), self.assignments_predicted)
            """ cls_loss: a scalar, of shape () """

        with tf.variable_scope('discriminator'):
            # Dsc for I1
            self.dsc_I_ref = self.discriminator(self.images_I_ref)
            """ dsc_I_ref: real/fake, of shape (64, 1) """
            # Dsc for I3
            self.dsc_I_ref_I_M_mix = self.discriminator(self.images_I_ref_I_M_mix, reuse=True)
            """ dsc_I_ref_I_M_mix: real/fake, of shape (64, 1) """

        with tf.variable_scope('discriminator_loss'):
            # Dsc loss x1
            self.dsc_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_I_ref), self.dsc_I_ref)
            # Dsc loss x3
            # this is max_D part of minmax loss function
            self.dsc_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.dsc_I_ref_I_M_mix), self.dsc_I_ref_I_M_mix)
            self.dsc_loss = self.dsc_loss_real + self.dsc_loss_fake
            """ dsc_loss: a scalar, of shape () """

        with tf.variable_scope('generator_loss'):
            # D (fix Dsc you have loss for G) -> cf. Dec
            # images_x3 = Dec(f_1_2) = G(f_1_2); Dsc(images_x3) = dsc_x3
            # rationale behind g_loss: this is min_G part of minmax loss function: min log D(G(x))
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_I_ref_I_M_mix), self.dsc_I_ref_I_M_mix)

        with tf.variable_scope('L2') as _:
            # Reconstruction loss L2 between I1 and I1' (to ensure autoencoder works properly)
            self.rec_loss_I_ref_hat_I_ref = tf.reduce_mean(tf.square(self.images_I_ref_hat - self.images_I_ref))
            """ rec_loss_x1hat_x1: a scalar, of shape () """
            # Reconstruction loss L2 between I2 and I2' (to ensure autoencoder works properly)
            self.rec_loss_I_M_hat_I_M = tf.reduce_mean(tf.square(self.images_I_M_hat - self.images_I_M))
            # L2 between I1 and I4
            self.rec_loss_I_ref_4_I_ref = tf.reduce_mean(tf.square(self.images_I_ref_4 - self.images_I_ref))
            # L2 between I2 and I5
            self.rec_loss_I_M_5_I_M = tf.reduce_mean(tf.square(self.images_I_M_5 - self.images_I_M))

        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()
        # Tf stuff (tell variables how to train..)
        self.dsc_vars = [var for var in t_vars if 'discriminator' in var.name and 'd_' in var.name] # discriminator
        self.gen_vars = [var for var in t_vars if 'generator' in var.name and 'g_' in var.name] # encoder + decoder (generator)
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

        g_loss_comp = 5 * self.rec_loss_I_ref_hat_I_ref + 5 * self.rec_loss_I_M_hat_I_M + 5 * self.rec_loss_I_ref_4_I_ref + 5 * self.rec_loss_I_M_5_I_M + 1 * self.g_loss + 1 * self.cls_loss
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

                if np.mod(iteration, 500) == 1:
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

    def path(self, filename):
        return os.path.join(self.params.summary_dir, filename)

    def discriminator(self, image, keep_prob=0.5, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # cf. DCGAN impl https://github.com/carpedm20/DCGAN-tensorflow.git
        h0 = lrelu(conv2d(image, self.df_dim, use_spectral_norm=True, name='d_1_h0_conv'))
        h1 = lrelu(conv2d(h0, self.df_dim*2, use_spectral_norm=True, name='d_1_h1_conv'))

        # #################################
        # ch = self.df_dim*2
        # x = h1
        # h1 = attention(x, ch, sn=True, scope="d_attention", reuse=reuse)
        # #################################

        h2 = lrelu(conv2d(h1, self.df_dim*4, use_spectral_norm=True, name='d_1_h2_conv'))
        # NB: k=1,d=1 is like an FC layer -> to strengthen h3, to give it more capacity
        h3 = lrelu(conv2d(h2, self.df_dim*8,k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='d_1_h3_conv'))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_1_h3_lin')

        return tf.nn.sigmoid(h4)


    def classifier(self, x1_tile1, x1_tile2, x1_tile3, x1_tile4,
                   x2_tile1, x2_tile2, x2_tile3, x2_tile4,
                   x3_tile1, x3_tile2, x3_tile3, x3_tile4, reuse=False):
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
        s5 = lrelu(instance_norm(conv2d(s4, self.df_dim * 8, k_h=1, k_w=1, use_spectral_norm=True, name='g_1_conv5')))
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

        # #################################
        # ch = self.gf_dim*4
        # x = h3
        # h3 = attention(x, ch, sn=True, scope="g_attention", reuse=reuse)
        # #################################

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
        tf.summary.scalar('rec_loss_I1hat_I1', self.rec_loss_I_ref_hat_I_ref)
        tf.summary.scalar('rec_loss_I2hat_I2', self.rec_loss_I_M_hat_I_M)
        tf.summary.scalar('rec_loss_I4_I1', self.rec_loss_I_ref_4_I_ref)
        tf.summary.scalar('rec_loss_I5_I2', self.rec_loss_I_M_5_I_M)


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
        images_Iref, images_IM, images_IrefImMixOut, \
        images_IrefImMixIn, \
        images_Iref4, images_IM5, \
        ass_actual = \
            self.sess.run([self.images_I_ref, self.images_I_M, self.images_I_ref_I_M_mix, \
                           self.images_J, \
                           self.images_I_ref_4, self.images_I_M_5, \
                           self.assignments_actual])
        grid_size = np.ceil(np.sqrt(self.batch_size))
        grid = [grid_size, grid_size]
        save_images(images_Iref, grid, self.path('%s_images_I_ref.jpg' % counter))
        save_images(images_IM, grid, self.path('%s_images_I_M.jpg' % counter))
        st = ''
        for list in ass_actual:
            for e in list:
                st += str(e)
            st += '_'
        st = st[:-1]
        file_path = self.path('%s_images_Iref_IM_mix_IN_%s.jpg' % (counter, st))
        save_images(images_IrefImMixIn, grid, file_path)
        file_path = self.path('%s_images_Iref_IM_mix_OUT_%s.jpg' % (counter, st))
        save_images(images_IrefImMixOut, grid, file_path)
        save_images(images_Iref4, grid, self.path('%s_images_I_ref_4.jpg' % counter))
        save_images(images_IM5, grid, self.path('%s_images_I_M_5.jpg' % counter))
