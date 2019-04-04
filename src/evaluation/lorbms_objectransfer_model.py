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
import tensorflow.contrib.slim as slim
from scipy.misc import imsave
import traceback
import csv
from random import randint
import cv2

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

        self.end = False

        self.random_seed = random_seed

        # exp74:
        self.useIRefAndMixForGanLoss = False
        self.lambda_mix = 0.50
        self.lambda_ref = 0.50

        self.build_model()


    def build_model(self):
        print("build_model() ------------------------------------------>")
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        tf.set_random_seed(self.random_seed)

        ####################################################################################
        # Load data
        ####################################################################################

        def _parse_function(file_Iref, file_Iobj):
            # print("_parse_function: " , ref_path)
            # image_string = tf.read_file(path_Iref[0])
            # file_Iref = tf.image.decode_jpeg(image_string, channels=3)
            # file_Iref = tf.image.decode_png(image_string, channels=3)
            #print("file_Iref: ", file_Iref)
            #file_Iref = crop_max(file_Iref)
            image_Iref_resized = tf.image.resize_images(file_Iref, [64, 64]) #, method=tf.image.ResizeMethod.AREA) for PNG?
            image_Iref_resized = tf.cast(image_Iref_resized, tf.float32) * (2. / 255) - 1

            #image_string = tf.read_file(path_Iobj[0])
            # file_Iobj = tf.image.decode_jpeg(image_string, channels=3)
            #file_Iobj = tf.image.decode_png(image_string, channels=3)
            #file_Iobj = crop_max(file_Iobj)
            image_Iobj_resized = tf.image.resize_images(file_Iobj, [64, 64]) # , method=tf.image.ResizeMethod.AREA) for PNG?
            image_Iobj_resized = tf.cast(image_Iobj_resized, tf.float32) * (2. / 255) - 1
            return image_Iref_resized, image_Iobj_resized

        # self.images_I_ref_plh = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        # self.images_I_obj_plh = tf.placeholder(tf.float32, shape=[1, None, None, 3])

        print("self.params.image_ref_path: %s" % self.params.image_ref_path)
        print("self.params.image_obj_path: %s" % self.params.image_obj_path)

        # = tf.constant([self.params.image_ref_path], dtype=tf.string)
        self.images_I_ref_plh = tf.placeholder(tf.float32, shape=[1, 412, 412, 3])
        # obj_path = tf.constant([self.params.image_obj_path], dtype=tf.string)
        self.images_I_obj_plh = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        #print(ref_path.shape)
        #print(obj_path.shape)
        dataset = tf.data.Dataset.from_tensor_slices((self.images_I_ref_plh, self.images_I_obj_plh))
        dataset = dataset.repeat().batch(self.batch_size)
        self.dataset = dataset.map(_parse_function)

        # self.iterator = dataset.make_one_shot_iterator()
        self.iterator = self.dataset.make_initializable_iterator()
        images_I_ref, images_I_obj = self.iterator.get_next() # Notice: for both train + test images!!
        images_I_ref = tf.reshape(images_I_ref, [self.batch_size, 64, 64, 3])
        images_I_obj = tf.reshape(images_I_obj, [self.batch_size, 64, 64, 3])
        print("images_I_ref: %s" % images_I_ref)
        print("images_I_obj: %s" % images_I_obj)
        # images_I_ref = tf.cast(images_I_ref, tf.float32) * (2. / 255) - 1
        # images_I_ref = tf.reshape(images_I_ref, (self.image_size, self.image_size, 3))
        # images_I_ref = tf.expand_dims(images_I_ref, 0)
        # images_I_obj = tf.cast(images_I_obj, tf.float32) * (2. / 255) - 1
        # images_I_obj = tf.reshape(images_I_obj, (self.image_size, self.image_size, 3))
        # images_I_obj = tf.expand_dims(images_I_obj, 0)

        print("images_I_ref: ", images_I_ref)
        print("images_I_obj: ", images_I_obj)
        self.images_I_ref = images_I_ref
        self.images_I_obj = images_I_obj

        self.feature_mix = [int(s) for s in self.params.feature_mix.split(',')]
        print("self.feature_mix: %s" % str(self.feature_mix))


        # ###########################################################################################################

        self.chunk_num = self.params.chunk_num
        """ number of chunks: 8 """
        self.chunk_size = self.params.chunk_size
        """ size per chunk: 64 """
        self.feature_size_tile = self.chunk_size * self.chunk_num
        """ equals the size of all chunks from a single tile """
        self.feature_size = self.feature_size_tile * NUM_TILES_L2_MIX
        """ equals the size of the full image feature """


        with tf.variable_scope('generator') as scope_generator:
            # params for ENCODER
            model = self.params.autoencoder_model
            coordConvLayer = True
            ####################

            self.I_ref_f = encoder_dense(self.images_I_ref, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)


            feature_tile_shape = [self.batch_size, self.feature_size_tile]
            self.I_ref_f1 = tf.slice(self.I_ref_f, [0, self.feature_size_tile * 0], feature_tile_shape)
            self.I_ref_f2 = tf.slice(self.I_ref_f, [0, self.feature_size_tile * 1], feature_tile_shape)
            self.I_ref_f3 = tf.slice(self.I_ref_f, [0, self.feature_size_tile * 2], feature_tile_shape)
            self.I_ref_f4 = tf.slice(self.I_ref_f, [0, self.feature_size_tile * 3], feature_tile_shape)


            # this is used to build up graph nodes (variables) -> for later reuse_variables..
            self.decoder(self.I_ref_f, preset_model=model, dropout_p=0.0)

            # to share the weights between the Encoders
            scope_generator.reuse_variables()

            self.I_obj_f = encoder_dense(self.images_I_obj, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)

            self.I_obj_f1 = tf.slice(self.I_obj_f, [0, self.feature_size_tile * 0], feature_tile_shape)
            self.I_obj_f2 = tf.slice(self.I_obj_f, [0, self.feature_size_tile * 1], feature_tile_shape)
            self.I_obj_f3 = tf.slice(self.I_obj_f, [0, self.feature_size_tile * 2], feature_tile_shape)
            self.I_obj_f4 = tf.slice(self.I_obj_f, [0, self.feature_size_tile * 3], feature_tile_shape)


            for id in range(self.batch_size):
                cond_q1 = tf.equal(self.feature_mix[0], FROM_I_OBJ)
                feature_1 = tf.expand_dims(tf.where(cond_q1, self.I_obj_f1[id], self.I_ref_f1[id]), 0)
                cond_q2 = tf.equal(self.feature_mix[1], FROM_I_OBJ)
                feature_2 = tf.expand_dims(tf.where(cond_q2, self.I_obj_f2[id], self.I_ref_f2[id]), 0)
                cond_q3 = tf.equal(self.feature_mix[2], FROM_I_OBJ)
                feature_3 = tf.expand_dims(tf.where(cond_q3, self.I_obj_f3[id], self.I_ref_f3[id]), 0)
                cond_q4 = tf.equal(self.feature_mix[3], FROM_I_OBJ)
                feature_4 = tf.expand_dims(tf.where(cond_q4, self.I_obj_f4[id], self.I_ref_f4[id]), 0)

                f_features_selected = tf.concat(axis=0, values=[feature_1, feature_2, feature_3, feature_4])  # axis=1
                f_features_selected = tf.reshape(f_features_selected, [-1])
                f_features_selected = tf.expand_dims(f_features_selected, 0)
                self.f_I_ref_I_M_mix = f_features_selected if id == 0 else tf.concat(axis=0, values=[self.f_I_ref_I_M_mix, f_features_selected])

            self.images_I_mix = self.decoder(self.f_I_ref_I_M_mix, preset_model=model, dropout_p=0.0)


        t_vars = tf.trainable_variables()
        self.gen_vars = [var for var in t_vars if 'generator' in var.name and 'g_' in var.name] # encoder + decoder (generator)
        self.print_model_params(t_vars)

        print("build_model() ------------------------------------------<")
        # END of build_model


    def test(self, params):
        """Train DCGAN"""

        t_vars = tf.trainable_variables()
        self.restore_autoencoder(params)

        self.initialize_uninitialized(tf.global_variables(), "global")
        self.initialize_uninitialized(tf.local_variables(), "local")

        print("initialize SN...")
        # in addition, for spectral normalization: initialize parameters u,v
        update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS)
        for update_op in update_ops:
            self.sess.run(update_op)

        self.print_model_params(t_vars)

        if params.continue_from:
            assert 1 == 0, "not supported"

        ##############################################################################################
        # TEST
        ##############################################################################################

        #with open(self.params.image_ref_path) as f:
        #    image_I_ref = np.fromfile(f, dtype=np.uint8, count=-1)
        # image_I_ref = imread(self.params.image_ref_path)
        # print("image_I_ref: %s", image_I_ref.shape)
        img = cv2.imread(self.params.image_ref_path)
        b, g, r = cv2.split(img) # get b,g,r channels
        image_I_ref = cv2.merge([r,g,b])
        print("image_I_ref: %s", image_I_ref.shape)
        image_I_ref = np.expand_dims(image_I_ref, 0)
        print("image_I_ref: %s", image_I_ref.shape)

        # with open(self.params.image_obj_path) as f:
        #    image_I_obj = np.fromfile(f, dtype=np.uint8, count=-1)
        # image_I_obj = imread(self.params.image_obj_path)
        # print("image_I_obj: %s", image_I_obj.shape)
        img = cv2.imread(self.params.image_obj_path)
        b, g, r = cv2.split(img) # get b,g,r channels
        image_I_obj = cv2.merge([r,g,b])
        print("image_I_obj: %s", image_I_obj.shape)
        image_I_obj = np.expand_dims(image_I_obj, 0)
        print("image_I_obj: %s", image_I_obj.shape)

        self.sess.run(self.iterator.initializer, feed_dict={self.images_I_ref_plh: image_I_ref, self.images_I_obj_plh: image_I_obj})

        ##############################################################################################

        imgs_I_ref, imgs_I_obj, imgs_I_mix = self.sess.run([self.images_I_ref, self.images_I_obj, self.images_I_mix])
        img_I_mix = imgs_I_mix[0]
        print("img_I_mix: ", img_I_mix.shape)

        img_I_mix_ass = to_string(self.feature_mix)
        fn = os.path.join(self.params.image_mix_out, img_I_mix_ass)
        if not os.path.exists(fn):
            os.makedirs(fn)
            print('created fn: %s' % fn)
        fn_mix = os.path.join(fn, "I_mix_" + img_I_mix_ass + ".png")
        fn_all = os.path.join(fn, "I_ref_I_obj_I_mix_" + img_I_mix_ass + ".png")

        imsave(fn_mix, img_I_mix)

        save_images_6cols(imgs_I_ref[0], imgs_I_obj[0], img_I_mix, None, None, None, [1,3], 1, fn_all, maxImg=1, addSpacing=4)

        print("saved image I_mix to %s." % fn)
        # END of test()


    def initialize_uninitialized(self, vars, context):
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in vars])
        not_initialized_vars = [v for (v, f) in zip(vars, is_not_initialized) if not f]

        print("#not initialized variables '%s': %d" % (context, len(not_initialized_vars))) # only for testing
        if len(not_initialized_vars):
            print("initializing not_initialized_vars: %s" % str(not_initialized_vars))
            self.sess.run(tf.variables_initializer(not_initialized_vars))


    def restore_autoencoder(self, params):
        ae_vars = [var.name for var in self.gen_vars]
        variables = slim.get_variables_to_restore(include=ae_vars)
        # print("variables1: ", variables)

        path = params.ae_checkpoint_name # if not self.isIdeRun else "../checkpoints/exp70/checkpoint/DCGAN.model-50"
        print('restoring auotencoder to [%s]...' % path)
        init_restore_op, init_feed_dict  = slim.assign_from_checkpoint(model_path=path, var_list=variables)
        self.sess.run(init_restore_op, feed_dict=init_feed_dict)
        print('autoencoder restored.')


    def path(self, filename):
        return os.path.join(self.params.summary_dir, filename)


    def decoder(self, inputs, preset_model, dropout_p=0.2):
        return decoder_dense(inputs, self.batch_size, self.feature_size, preset_model=preset_model, dropout_p=dropout_p)

    def handle_exit(self, signum, frame):
        self.end = True

    def print_model_params(self, t_vars):
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
    for e in ass_actual:
        st += str(e)
        if elem_sep:
            st += elem_sep
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

def crop_max(file_Iref):
    height = file_Iref.shape[0]
    width = file_Iref.shape[1]
    size = tf.minimum(height, width)
    crop_shape = tf.parallel_stack([size, size, 3])
    file_Iref = tf.random_crop(file_Iref, crop_shape, seed=4285)
    return file_Iref

