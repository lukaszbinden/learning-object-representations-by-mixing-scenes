import signal
from ops_alex import *
from utils_dcgan import *
from utils_common import *
# from util_densenet import encoder_dense, decoder_dense
# from input_pipeline_rendered_data import get_pipeline_training_from_dump
from input_pipeline import *
from constants import *
import socket
import numpy as np
import traceback
from autoencoder_dblocks import encoder_dense, decoder_dense
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
        self.batch_size_cherry = 16 # fixed
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
        # self.crop_shape = tf.parallel_stack([self.batch_size, image_size // 2, image_size // 2, 3])
        # self.crop_shape = [self.batch_size, image_size // 2, image_size // 2, 3]

        isIdeRun = 'lz826' in os.path.realpath(sys.argv[0])
        file_train = 'datasets/coco/2017_training/version/v1/final/' if not isIdeRun else 'data/train-00011-of-00060.tfrecords'
        file_test = 'datasets/coco/2017_val/version/v1/final/' if not isIdeRun else 'data/train-00011-of-00060.tfrecords'
        file_test_cherry = 'datasets/coco/2017_val/version/v4/final/' if not isIdeRun else 'data/train-00011-of-00060.tfrecords'

        cwd = os.getcwd()
        file_train = os.path.join(cwd, file_train)
        file_test = os.path.join(cwd, file_test)
        file_test_cherry = os.path.join(cwd, file_test_cherry)

        ####################################################################################
        reader = tf.TFRecordReader()
        rrm_fn = lambda name : read_record_max(name, reader, image_size)
        _, train_images, _, _, _, _, _, _, _, _, _, _, _, _ = \
                get_pipeline(file_train, self.batch_size, self.epochs, rrm_fn)
        print('train_images.shape..:', train_images.shape)
        self.images_I_ref = train_images
        # self.images_I_ref_crop = tf.random_crop(self.images_I_ref, self.crop_shape, seed=4285)


        reader_test = tf.TFRecordReader()
        rrm_fn = lambda name: read_record_max(name, reader_test, image_size)
        _, test_images, _, _, _, _, _, _, _, _, _, _, _, _ = \
            get_pipeline(file_test, self.batch_size, self.epochs * 10000, rrm_fn)
        print('test_images.shape..:', test_images.shape)
        self.images_I_test = test_images

        reader_test = tf.TFRecordReader()
        rrm_fn = lambda name: read_record_max(name, reader_test, image_size, crop=False)
        _, test_images_cherry, _, _, _, _, _, _, _, _, _, _, _, _ = \
            get_pipeline_cherry(file_test_cherry, self.batch_size_cherry, self.epochs * 100000, rrm_fn)
        print('test_images_cherry.shape..:', test_images_cherry.shape)
        self.images_I_test_cherry = test_images_cherry


        self.chunk_num = self.params.chunk_num
        """ number of chunks: 8 """
        self.chunk_size = self.params.chunk_size
        """ size per chunk: 64 """
        self.feature_size = self.chunk_size*self.chunk_num
        """ equals the size of all chunks from a single tile """

        with tf.variable_scope('generator') as scope_generator:

            model = 'FC-DenseNet103'
            # TODO: add spectral norm!
            self.I_ref_f = encoder_dense(self.images_I_ref, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model)
            # self.I_ref_f = self.encoder(self.images_I_ref, self.batch_size)
            assert self.I_ref_f.shape[0] == self.batch_size
            assert self.I_ref_f.shape[1] == self.feature_size

            # this is used to build up graph nodes (variables) -> for later reuse_variables..
            #self.decoder(self.f_I_ref_composite)
            self.images_I_ref_hat = decoder_dense(self.I_ref_f, self.batch_size, self.feature_size, preset_model=model, dropout_p=0.0, reuse=False)
            # self.images_I_ref_hat_crop = tf.random_crop(self.images_I_ref_hat, self.crop_shape, seed=4285)
            # self.images_I_ref_hat = self.decoder(self.I_ref_f, self.batch_size)

            # to share the weights between the Encoders
            scope_generator.reuse_variables()

            self.I_test_f = encoder_dense(self.images_I_test, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model)
            self.images_I_test_hat = decoder_dense(self.I_test_f, self.batch_size, self.feature_size, preset_model=model, dropout_p=0.0, reuse=True)
            # self.I_test_f = self.encoder(self.images_I_test, self.batch_size)
            # self.images_I_test_hat = self.decoder(self.I_test_f, self.batch_size)

            self.I_test_f_cherry = encoder_dense(self.images_I_test_cherry, self.batch_size_cherry, self.feature_size, dropout_p=0.0, preset_model=model)
            self.images_I_test_hat_cherry = decoder_dense(self.I_test_f_cherry, self.batch_size_cherry, self.feature_size, preset_model=model, dropout_p=0.0, reuse=True)
            # self.I_test_f_cherry = self.encoder(self.images_I_test_cherry, self.batch_size_cherry)
            # self.images_I_test_hat_cherry = self.decoder(self.I_test_f_cherry, self.batch_size_cherry)

            self.images_I_ref_psnr = tf.reduce_mean(tf.image.psnr(self.images_I_ref, self.images_I_ref_hat, max_val=1.0))
            self.images_I_test_psnr = tf.reduce_mean(tf.image.psnr(self.images_I_test, self.images_I_test_hat, max_val=1.0))
            self.images_I_test_cherry_psnr = tf.reduce_mean(tf.image.psnr(self.images_I_test_cherry, self.images_I_test_hat_cherry, max_val=1.0))


        with tf.variable_scope('discriminator'):
            # Dsc for I1
            self.dsc_I_ref = self.discriminator(self.images_I_ref)
            # self.dsc_I_ref = self.discriminator_global_local(self.images_I_ref, self.images_I_ref_crop)
            # assert self.dsc_I_reftmp.shape == self.dsc_I_ref.shape

            """ dsc_I_ref: real/fake, of shape (64, 1) """
            # Dsc for I3
            self.dsc_I_ref_hat = self.discriminator(self.images_I_ref_hat, reuse=True)
            # self.dsc_I_ref_hat = self.discriminator_global_local(self.images_I_ref_hat, self.images_I_ref_hat_crop, reuse=True)

            # just for logging purposes:
            self.dsc_I_ref_mean = tf.reduce_mean(self.dsc_I_ref)
            self.dsc_I_ref_hat_mean = tf.reduce_mean(self.dsc_I_ref_hat)
            self.v_g_d = tf.reduce_mean(tf.log(self.dsc_I_ref) + tf.log(1 - self.dsc_I_ref_hat))

            """ dsc_I_ref_I_M_mix: real/fake, of shape (64, 1) """

        with tf.variable_scope('discriminator_loss'):
            # Dsc loss x1
            self.dsc_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_I_ref), self.dsc_I_ref)
            # Dsc loss x3
            # this is max_D part of minmax loss function
            self.dsc_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.dsc_I_ref_hat), self.dsc_I_ref_hat)
            self.dsc_loss = self.dsc_loss_real + self.dsc_loss_fake
            """ dsc_loss: a scalar, of shape () """

        with tf.variable_scope('generator_loss'):
            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.dsc_I_ref_hat), self.dsc_I_ref_hat)

        with tf.variable_scope('L1') as _:
            # Reconstruction loss L2 between I1 and I1' (to ensure autoencoder works properly)
            self.rec_loss_I_ref_hat_I_ref = tf.reduce_mean(tf.abs(self.images_I_ref_hat - self.images_I_ref))


        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()
        # Tf stuff (tell variables how to train..)
        self.dsc_vars = [var for var in t_vars if 'discriminator' in var.name and 'd_' in var.name] # discriminator
        self.gen_vars = [var for var in t_vars if 'generator' in var.name and 'g_' in var.name] # encoder + decoder (generator)
        # self.cls_vars = [var for var in t_vars if 'c_' in var.name] # classifier

        self.print_model_params(t_vars)

        # save the weights
        self.saver = tf.train.Saver(self.dsc_vars + self.gen_vars + batch_norm.shadow_variables, max_to_keep=5)
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


        print('g_learning_rate: %s' % self.g_learning_rate)
        print('d_learning_rate: %s' % self.d_learning_rate)

        # g_loss_comp = 5 * self.rec_loss_I_ref_hat_I_ref + 5 * self.rec_loss_I_M_hat_I_M + 5 * self.rec_loss_I_ref_4_I_ref + 5 * self.rec_loss_I_M_5_I_M + 1 * self.g_loss + 1 * self.cls_loss
        # g_loss_comp = 40 * self.rec_loss_I_ref_hat_I_ref + 1 * self.g_loss
        g_loss_comp = 0.998 * self.rec_loss_I_ref_hat_I_ref + 0.002 * self.g_loss

        # for autoencoder
        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=params.beta1, beta2=params.beta2) \
                          .minimize(g_loss_comp, var_list=self.gen_vars) # includes encoder + decoder weights

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
                # TODO as in DCGAN-tensorflow, perhabs run g_optim twice? "to make sure that d_loss does not go to zero (different from paper)"

                # Update D and G network
                self.sess.run([g_optim])
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

        h2 = lrelu(conv2d(h1, self.df_dim*4, use_spectral_norm=True, name='d_1_h2_conv'))

        #################################
        ch = self.df_dim*4
        x = h2
        h2 = attention(x, ch, sn=True, scope="d_attention", reuse=reuse)
        #################################

        h3 = lrelu(conv2d(h2, self.df_dim * 8, use_spectral_norm=True, name='d_1_h3_conv'))

        # NB: k=1,d=1 is like an FC layer -> to strengthen h3, to give it more capacity
        h3 = lrelu(conv2d(h3, self.df_dim*8,k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='d_1_h4_conv'))
        print('h3 resolution before FC:', h3.shape)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, use_spectral_norm=True, name='d_1_h4_lin')

        return tf.nn.sigmoid(h4)


    def discriminator_global_local(self, image, crop, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # TODO: evtl. use SA layer as well?

        # global 64
        c0_0 = lrelu(conv2d(image, 64, k_h=3, k_w=3, d_h=1, d_w=1, use_spectral_norm=True, name='d_c0_0'))
        c0_1 = lrelu(conv2d(c0_0, 128, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_c0_1'))
        # c1_0 = lrelu(conv2d(c0_1, 128, k_h=3, k_w=3, d_h=1, d_w=1, use_spectral_norm=True, name='d_c1_0'))
        c1_1 = lrelu(conv2d(c0_1, 256, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_c1_1'))
        # c2_0 = lrelu(conv2d(c1_1, 256, k_h=3, k_w=3, d_h=1, d_w=1, use_spectral_norm=True, name='d_c2_0'))
        c2_1 = lrelu(conv2d(c1_1, 256, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_c2_1'))
        c3_0 = lrelu(conv2d(c2_1, 256, k_h=3, k_w=3, d_h=1, d_w=1, use_spectral_norm=True, name='d_c3_0'))
        print('before reshape c3_0:', c3_0.shape)
        c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
        print('after reshape c3_0:', c3_0.shape)

        # c0_0 = sn_lrelu(sn_conv2d(image, 64, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c0_0'))
        # c0_1 = sn_lrelu(sn_conv2d(c0_0, 128, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c0_1'))
        # c1_0 = sn_lrelu(sn_conv2d(c0_1, 128, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c1_0'))
        # c1_1 = sn_lrelu(sn_conv2d(c1_0, 256, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c1_1'))
        # c2_0 = sn_lrelu(sn_conv2d(c1_1, 256, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c2_0'))
        # c2_1 = sn_lrelu(sn_conv2d(c2_0, 512, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c2_1'))
        # c3_0 = sn_lrelu(sn_conv2d(c2_1, 512, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c3_0'))
        # c3_0 = tf.reshape(c3_0, [self.batch_size, -1])

        # local 32
        y_l = lrelu(conv2d(crop, 32, k_h=3, k_w=3, d_h=1, d_w=1, use_spectral_norm=True, name='d_l0_0'))
        y_l = lrelu(conv2d(y_l, 64, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_l0_1'))
        # y_l = lrelu(conv2d(y_l, 64, k_h=3, k_w=3, d_h=1, d_w=1, use_spectral_norm=True, name='d_l1_0'))
        y_l = lrelu(conv2d(y_l, 128, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_l1_1'))
        y_l = lrelu(conv2d(y_l, 160, k_h=3, k_w=3, d_h=1, d_w=1, use_spectral_norm=True, name='d_l2_0'))
        print('before reshape: y_l', y_l.shape)
        y_l = tf.reshape(y_l, [self.batch_size, -1])
        print('after reshape: y_l', y_l.shape)

        y = tf.concat(values=[c3_0, y_l], axis=-1)

        print('y resolution before linear:', y.shape)
        y = linear(y, 1, stddev=0.01, use_spectral_norm=True,name='d_l4')
        print('y shape after linear:', y.shape)

        return tf.nn.sigmoid(y)


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

        # Comment 64: because of img size 64 I had to change this max_pool here..
        # --> undo this as soon as size 128 is used again...
        assert x1_tile1.shape[1] == 64
        # pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='c_3_pool5')
        # reduces size from (32, 2, 2, 256) to (32, 1, 1, 256)
        pool5 = max_pool(conv5, 2, 2, 1, 1, padding='VALID', name='c_3_pool5')

        fc6 = tf.nn.relu(linear(tf.reshape(pool5, [self.batch_size, -1]), 4096, name='c_3_fc6') )

        fc7 = tf.nn.relu(linear(tf.reshape(fc6, [self.batch_size, -1]), 4096, name='c_3_fc7') )

        self.fc8 = linear(tf.reshape(fc7, [self.batch_size, -1]), NUM_TILES_L2_MIX, name='c_3_fc8')

        return tf.nn.sigmoid(self.fc8)


    def encoder(self, tile_image, batch_size, reuse=False):
        return self.encoder_linear(tile_image, batch_size)


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

        print('rep before:', rep.shape)
        rep = tf.reshape(rep, [self.batch_size, -1])
        print('rep after:', rep.shape)
        assert rep.shape[0] == self.batch_size
        assert rep.shape[1] == self.feature_size

        return rep


    def encoder_linear(self, image, batch_size, reuse=False):
        """
        returns: 1D vector f1 with size=self.feature_size
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s0 = lrelu(instance_norm(conv2d(image, self.df_dim, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, self.df_dim * 2, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, self.df_dim * 4, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv2')))
        s3 = lrelu(instance_norm(conv2d(s2, self.df_dim * 8, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv3')))
        s4 = lrelu(instance_norm(conv2d(s3, self.df_dim * 12, k_h=2, k_w=2, d_h=2, d_w=2, use_spectral_norm=True, name='g_1_conv4')))

        s5 = lrelu(instance_norm(conv2d(s4, self.df_dim * 12, k_h=2, k_w=2, d_h=1, d_w=1, use_spectral_norm=True, name='g_1_conv5')))

        # exp21: adding an extra 1x1 conv layer s6 for more capacity...
        s6 = lrelu(instance_norm(conv2d(s5, self.df_dim * 12, k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='g_1_conv6')))

        print('s6:', s6.shape)

        # TODO Qiyang: why linear layer here?
        rep = lrelu((linear(tf.reshape(s6, [batch_size, -1]), self.feature_size, use_spectral_norm=True, name='g_1_fc')))
        print('rep:', rep.shape)

        assert rep.shape[0] == batch_size
        assert rep.shape[1] == self.feature_size

        return rep


    def decoder(self, representations, batch_size, reuse=False):
        """
        returns: batch of images with size 256x60x60x3
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshape = tf.reshape(representations,[batch_size, 1, 1, self.feature_size])

        h = deconv2d(reshape, [batch_size, 4, 4, self.gf_dim*4], k_h=4, k_w=4, d_h=1, d_w=1, padding='VALID', use_spectral_norm=True, name='g_de_h')
        h = tf.nn.relu(h)

        h1 = deconv2d(h, [batch_size, 8, 8, self.gf_dim*4], use_spectral_norm=True, name='g_h1')
        h1 = tf.nn.relu(instance_norm(h1))

        h2 = deconv2d(h1, [batch_size, 16, 16, self.gf_dim*2], use_spectral_norm=True, name='g_h2')
        h2 = tf.nn.relu(instance_norm(h2))

        h3 = deconv2d(h2, [batch_size, 32, 32, self.gf_dim*1], use_spectral_norm=True, name='g_h3')
        h3 = tf.nn.relu(instance_norm(h3))

        # #################################
        # ch = self.gf_dim*4
        # x = h3
        # h3 = attention(x, ch, sn=True, scope="g_attention", reuse=reuse)
        # #################################

        # h4 = deconv2d(h3, [self.batch_size, 64, 64, self.gf_dim*1], use_spectral_norm=True, name='g_h4')
        # h4 = tf.nn.relu(instance_norm(h4))

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
        h5 = h3
        h6 = deconv2d(h5, [batch_size, 64, 64, self.c_dim], use_spectral_norm=True, name='g_h7')

        return tf.nn.tanh(h6)


    def make_summary_ops(self, g_loss_comp):
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('g_loss_comp', g_loss_comp)
        tf.summary.scalar('dsc_loss', self.dsc_loss)
        tf.summary.scalar('dsc_loss_fake', self.dsc_loss_fake)
        tf.summary.scalar('dsc_loss_real', self.dsc_loss_real)
        tf.summary.scalar('rec_loss_Iref_hat_I_ref', self.rec_loss_I_ref_hat_I_ref)
        tf.summary.scalar('images_I_ref_psnr', self.images_I_ref_psnr)
        tf.summary.scalar('images_I_test_psnr', self.images_I_test_psnr)
        tf.summary.scalar('images_I_test_cherry_psnr', self.images_I_test_cherry_psnr)
        tf.summary.scalar('dsc_I_ref_mean', self.dsc_I_ref_mean)
        tf.summary.scalar('dsc_I_ref_hat_mean', self.dsc_I_ref_hat_mean)
        tf.summary.scalar('V_G_D', self.v_g_d)
        tf.summary.image('images_I_test_hat', self.images_I_test_hat)


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
        images_Iref, imgs_IrefHat, imgs_Itest, imgs_ItestHat, imgs_Itest_cherry, imgs_ItestHat_cherry, imgs_ref_psnr, imgs_test_psnr, imgs_cherry_psnr = \
            self.sess.run([self.images_I_ref, self.images_I_ref_hat, self.images_I_test, self.images_I_test_hat, \
                           self.images_I_test_cherry, self.images_I_test_hat_cherry, self.images_I_ref_psnr, self.images_I_test_psnr, self.images_I_test_cherry_psnr])

        # grid_size = np.ceil(np.sqrt(self.batch_size))
        # grid = [grid_size, grid_size]
        # save_images(images_Iref, grid, self.path('%s_images_I_ref.jpg' % counter))
        # save_images(imgs_IrefHat, grid, self.path('%s_images_I_ref_hat.jpg' % counter))
        # save_images(imgs_Itest, grid, self.path('%s_images_I_test.jpg' % counter))
        # save_images(imgs_ItestHat, grid, self.path('%s_images_I_test_hat.jpg' % counter))

        # grid = [4, 4] # fixed batch size of 16 cf self.batch_size_cherry
        # save_images(imgs_Itest_cherry, grid, self.path('%s_images_I_test_cherry.jpg' % counter))
        # save_images(imgs_ItestHat_cherry, grid, self.path('%s_images_I_test_hat_cherry.jpg' % counter))

        grid_size = min(self.batch_size, 16)
        grid = [grid_size, 2]
        save_images_multi(images_Iref, imgs_IrefHat, None, grid, self.batch_size, self.path('%s_images_I_ref_and_hat_%s.jpg' % (counter, str(round(imgs_ref_psnr, 2)))), maxImg=grid_size)
        save_images_multi(imgs_Itest, imgs_ItestHat, None, grid, self.batch_size, self.path('%s_images_I_test_and_hat_%s.jpg' % (counter, str(round(imgs_test_psnr, 2)))), maxImg=grid_size)
        grid = [self.batch_size_cherry, 2]
        save_images_multi(imgs_Itest_cherry, imgs_ItestHat_cherry, None, grid, self.batch_size_cherry, self.path('%s_images_I_test_and_hat_cherry_%s.jpg' % (counter, str(round(imgs_cherry_psnr, 2)))), maxImg=self.batch_size_cherry)
        print('var test cherry:', str(np.var(imgs_Itest_cherry)))
        print('var test hat cherry:', str(np.var(imgs_ItestHat_cherry)))


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
        # count_model_params(self.cls_vars, 'Classifier')
        count_model_params(t_vars, 'Total')


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

