from ops_alex import *
from utils_dcgan import *
from utils_common import *
from input_pipeline import *
from autoencoder_dblocks import encoder_dense
from patch_gan_discriminator_linearcls import Deep_PatchGAN_Discrminator
from constants import *
from alexnet import alexnet_conv1_conv5
import numpy as np
# from scipy.misc import imsave
# import traceback
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import collections
from Preprocessor import Preprocessor
# import scipy.misc

class DCGAN(object):
    img_preprocessor = None

    def __init__(self, sess, params,
                 batch_size=256, sample_size = 64, epochs=1000, image_shape=[256, 256, 3],
                 y_dim=None, z_dim=0, gf_dim=128, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1,
                 is_train=True, random_seed=4285):
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

        self.isIdeRun = 'lz826' in os.path.realpath(sys.argv[0])

        self.isTraining = True

        target_shape = [self.image_size, self.image_size, 3]
        DCGAN.img_preprocessor = Preprocessor(target_shape=target_shape)

        self.build_model()


    def build_model(self):
        print("build_model() ------------------------------------------>")
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        tf.set_random_seed(self.random_seed)

        # image_size = self.image_size
        self.feature_size_tile = self.params.chunk_size * self.params.chunk_num
        self.feature_size = self.feature_size_tile * NUM_TILES_L2_MIX

        ########################### STL-10 BEGIN

        def _parse(image, label):
            # as in [70]: "We randomly resize the images and extract 96 x 96 crops"
            # => randomly resize and extract 64 x 64 crops

            # the image augmentation is analogous to S. Jennis code in paper "Self-Supervised Feature Learning by Learning to Spot Artifacts"
            image_processed = DCGAN.img_preprocessor.process(image)

            return image_processed, label

        self.images_plh = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
        self.labels_plh = tf.placeholder(tf.int32, shape=[None])
        dataset = tf.data.Dataset.from_tensor_slices((self.images_plh, self.labels_plh)).repeat().shuffle(self.batch_size).batch(self.batch_size)
        self.dataset = dataset.map(_parse)

        self.iterator = self.dataset.make_initializable_iterator()
        images, labels = self.iterator.get_next() # Notice: for both train + test images!!
        print("************************************", images)
        images = tf.reshape(images, [self.batch_size, self.image_size, self.image_size, 3])
        print("images: ", images)
        print("labels: ", labels)
        y = tf.one_hot(labels, 10, dtype=tf.int32)
        y_onehot = tf.reshape(y, [self.batch_size, 10])
        ########################### STL-10 END

        self.images_I_ref = images
        self.labels = labels
        self.labels_onehot = y_onehot


        if self.params.encoder_type == 'alexnet':
            with tf.variable_scope('alexnet'):
                self.I_ref_f = self.alexnet(self.images_I_ref)

        elif self.params.encoder_type == 'encoder':
            with tf.variable_scope('generator'):
                model = self.params.autoencoder_model
                coordConvLayer = True
                ####################
                print("using encoder for TL...")
                self.I_ref_f = encoder_dense(self.images_I_ref, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)
        else:
            assert 1 == 0, self.params.encoder_type + " not supported!"

        with tf.variable_scope('classifier'):
            print("self.I_ref_f: ", self.I_ref_f.shape)
            self.lin_cls_logits = self.linear_classifier(self.I_ref_f)


        with tf.variable_scope('classifier_loss'):
            self.cls_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_onehot, logits=self.lin_cls_logits, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            tf.summary.scalar('cls_loss', self.cls_loss)


        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(self.lin_cls_logits, 1), tf.argmax(self.labels_onehot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        t_vars = tf.trainable_variables()
        save_vars = []

        if self.params.encoder_type == 'alexnet':
            self.enc_vars = [var for var in t_vars if 'alexnet' in var.name] # just alexnet
            self.gen_vars = []
            save_vars.extend(self.enc_vars)

        elif self.params.encoder_type == 'encoder':
            self.gen_vars = [var for var in t_vars if 'generator' in var.name and 'g_' in var.name] # encoder (generator)
            print("gen_vars:", self.gen_vars)
            self.enc_vars = []
            save_vars.extend(self.gen_vars)

        else:
            assert 1 == 0, self.params.encoder_type + " not supported!"

        self.cls_vars = [var for var in t_vars if 'classifier' in var.name]
        save_vars.extend(self.cls_vars)

        list = []
        list.extend(self.gen_vars)
        list.extend(self.cls_vars)
        list.extend(self.enc_vars)
        assert collections.Counter(list) == collections.Counter(t_vars)
        assert collections.Counter(save_vars) == collections.Counter(t_vars)
        del list

        print("parameters after print_model_params: ****************")
        self.print_model_params(t_vars)
        print("*****************************************************")

        # save encoder_type + CLS
        self.saver = tf.train.Saver(save_vars, max_to_keep=5)
        if self.params.is_train:
            self.saver_metrics = tf.train.Saver(save_vars, max_to_keep=None)
        print("build_model() ------------------------------------------<")
        # END of build_model


    def train(self, params):
        """Train DCGAN"""

        iteration = 0

        global_step = tf.Variable(iteration, name='global_step', trainable=False)

        cls_learning_rate = tf.constant(params.learning_rate_cls)
        # cls_learning_rate = tf.train.exponential_decay(cls_learning_rate, global_step=global_step,
        #                                                   decay_steps=800, decay_rate=0.9, staircase=True)
        print('cls_learning_rate: %s' % cls_learning_rate)

        _, acc_update_op = tf.metrics.accuracy(labels=tf.argmax(self.labels_onehot, axis=1), predictions=tf.argmax(self.lin_cls_logits, axis=1, output_type=tf.int32))
        tf.summary.scalar('acc_update_op', acc_update_op)
        _, test_acc_update_op = tf.metrics.accuracy(labels=tf.argmax(self.labels_onehot, axis=1), predictions=tf.argmax(self.lin_cls_logits, axis=1, output_type=tf.int32))
        tf.summary.scalar('test_acc_update_op', test_acc_update_op)

        t_vars = tf.trainable_variables()
        # restore encoder from checkpoint
        print("params.encoder_type: %s" % params.encoder_type)
        if params.encoder_type == "encoder":
            self.gen_vars = []
            self.cls_vars = t_vars # use all vars incl. generator for training

        elif params.encoder_type == "alexnet":
            assert len(self.gen_vars) == 0
            self.enc_vars = []
            self.cls_vars = t_vars # use all vars incl. encoder for training

        else:
            assert 1 == 0, self.params.encoder_type + " not supported!"

        # for classifier
        c_optim = tf.train.AdamOptimizer(learning_rate=cls_learning_rate, beta1=0.5) \
                          .minimize(self.cls_loss, var_list=self.cls_vars, global_step=global_step)  # params.beta1

        self.initialize_uninitialized(tf.global_variables(), "global")
        self.initialize_uninitialized(tf.local_variables(), "local")

        self.print_model_params(t_vars)

        ##############################################################################################
        # LOAD DATA SET
        ##############################################################################################

        dataset_path = self.params.dataset_path if not self.isIdeRun else 'D:\\learning-object-representations-by-mixing-scenes\\src\\datasets\\stl-10\\stl10_binary'
        DATA_DIR = dataset_path

        with open(os.path.join(DATA_DIR, 'train_X.bin')) as f:
            raw = np.fromfile(f, dtype=np.uint8, count=-1)
            raw = np.reshape(raw, (-1, 3, 96, 96))
            raw = np.transpose(raw, (0, 3, 2, 1))
            X_train_raw = raw
            print("X_train_raw.shape: ", X_train_raw.shape)


        with open(os.path.join(DATA_DIR, 'train_y.bin')) as f:
            raw = np.fromfile(f, dtype=np.uint8, count=-1)
            raw = raw
            y_train = raw - 1  # class labels are originally in 1-10 format. Convert them to 0-9 format
            print("y_train.shape: ", y_train.shape)


        with open(os.path.join(DATA_DIR, 'test_X.bin')) as f:
            raw = np.fromfile(f, dtype=np.uint8, count=-1)
            raw = np.reshape(raw, (-1, 3, 96, 96))
            raw = np.transpose(raw, (0, 3, 2, 1))
            X_test_raw = raw


        with open(os.path.join(DATA_DIR, 'test_y.bin')) as f:
            raw = np.fromfile(f, dtype=np.uint8, count=-1)
            y_test = raw - 1

        ##############################################################################################
        # TRAINING
        ##############################################################################################

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(params.summary_dir)
        summary_writer.add_graph(self.sess.graph)

        self.sess.run(self.iterator.initializer, feed_dict={self.images_plh: X_train_raw, self.labels_plh: y_train,
                                                            DCGAN.img_preprocessor.training_mode_plh: True, DCGAN.img_preprocessor.augment_color_plh: True})

        n_batches = X_train_raw.shape[0] // self.batch_size + 1
        print("n_batches: %s, X_train_raw.shape[0]: %s, self.batch_size: %s" % (str(n_batches), str(X_train_raw.shape[0]), str(self.batch_size)))

        sum_train_loss_results = []
        sum_train_accuracy_results = []

        for i in range(params.epochs):
            train_loss_results = []
            train_accuracy_results = []
            for _ in range(n_batches):
                summary_str, _, loss_value, acc_value, gl, lr = self.sess.run([summary_op, c_optim, self.cls_loss, acc_update_op, global_step, cls_learning_rate])
                summary_writer.add_summary(summary_str, gl)
                train_loss_results.append(loss_value)
                sum_train_loss_results.append(loss_value)
                train_accuracy_results.append(acc_value)
                sum_train_accuracy_results.append(acc_value)
            print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}, Global step: {}, LR: {}".format(i + 1, np.mean(train_loss_results), np.mean(train_accuracy_results), str(gl), str(lr)))


        self.save(params.checkpoint_dir, gl)  # save model

        doPlot = False
        if doPlot:
            fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
            fig.suptitle('Training Metrics')
            axes[0].set_ylabel("Loss", fontsize=14)
            axes[0].plot(sum_train_loss_results)
            axes[1].set_ylabel("Accuracy", fontsize=14)
            axes[1].set_xlabel("Epoch", fontsize=14)
            axes[1].plot(sum_train_accuracy_results)
            plt.show()


        ##############################################################################################
        # TEST
        ##############################################################################################

        # the model only evaluates a single epoch of the test data
        self.sess.run(self.iterator.initializer, feed_dict={self.images_plh: X_test_raw, self.labels_plh: y_test,
                                                            DCGAN.img_preprocessor.training_mode_plh: False, DCGAN.img_preprocessor.augment_color_plh: False})
        test_loss_results = []
        test_accuracy_results = []

        n_batches = X_test_raw.shape[0] // self.batch_size + 1
        print("n_batches: %s, X_test_raw.shape[0]: %s, self.batch_size: %s" % (str(n_batches), str(X_test_raw.shape[0]), str(self.batch_size)))

        for b in range(n_batches):
            loss_value, acc_value, logits = self.sess.run([self.cls_loss, test_acc_update_op, self.lin_cls_logits])
            print("test_acc_value batch %d: %s, logits: %s" % (b+1, str(acc_value), str(logits.shape)))
            test_loss_results.append(loss_value)
            test_accuracy_results.append(acc_value)

        test_accuracy = np.mean(test_accuracy_results)
        test_std = np.std(test_accuracy_results)
        print("Test loss: avg: %f, std: %f" % (np.mean(test_loss_results), np.std(test_loss_results)))
        print("Test acc.: avg: %f, std: %f" % (test_accuracy, test_std))

        # END of train()


    def initialize_uninitialized(self, vars, context):
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in vars])
        not_initialized_vars = [v for (v, f) in zip(vars, is_not_initialized) if not f]

        print("#not initialized variables '%s': %d" % (context, len(not_initialized_vars))) # only for testing
        if len(not_initialized_vars):
            print("initializing not_initialized_vars: %s" % str(not_initialized_vars))
            self.sess.run(tf.variables_initializer(not_initialized_vars))


    def path(self, filename):
        return os.path.join(self.params.summary_dir, filename)

    def linear_classifier(self, features):
        features = tf.reshape(features, [self.batch_size, -1])
        print("features: ", features.shape)
        logits = tf.layers.dense(inputs=features, units=self.params.number_of_classes, use_bias=True, activation=None,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02, seed=4285),
                                 bias_initializer=tf.constant_initializer(0.02),
                                 name='Linear')
        return logits

    def alexnet(self, images):
        # LZ: for fair comparison with LORBMS encoder we only train conv1-conv5 with a linear classifier
        # instead of the entire AlexNet
        return alexnet_conv1_conv5(images)
        # return alexnet_v2(images, is_training=is_training)


    def discriminator(self, image, keep_prob=0.5, reuse=False, y=None):
        assert not self.params.discriminator_coordconv
        # if self.params.discriminator_coordconv:
        #     return self.discriminator_coordconv(image, keep_prob, reuse, y)

        if self.params.discriminator_patchgan:
            return self.discriminator_patchgan(image, reuse)

        _, h3 = self.discriminator_std(image, keep_prob, reuse, y, returnH3=True)
        return h3

    def discriminator_std(self, image, keep_prob=0.5, reuse=False, y=None, returnH3=False):
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
        # print("h3.shape: %s" % str(h3.shape)) # h3.shape: (128, 4, 4, 512)
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, use_spectral_norm=True, name='d_1_h4_lin')
        # print("h4.shape: %s" % str(h4.shape)) # h4.shape: (128, 1)

        if returnH3:
            return h4, h3

        # return tf.nn.sigmoid(h4)
        return h4

    def discriminator_patchgan(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        dsc = Deep_PatchGAN_Discrminator(addCoordConv=False, returnH4=True)

        res = dsc(image)
        # print('dsc: ', res.shape)

        return res

    def restore_encoder(self, params):
        enc_vars = [var.name for var in self.gen_vars if 'g_1' in var.name]
        variables = slim.get_variables_to_restore(include=enc_vars)
        # print("variables1: ", variables)

        path = params.encoder_checkpoint_name if not self.isIdeRun else "../checkpoints/exp70/checkpoint/DCGAN.model-50"
        print('restoring encoder to [%s]...' % path)
        init_restore_op, init_feed_dict  = slim.assign_from_checkpoint(model_path=path, var_list=variables)
        self.sess.run(init_restore_op, feed_dict=init_feed_dict)
        print('encoder restored.')

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.model_name)
        get_pp().pprint('[1] Save model to {} with step={}'.format(path, step))
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
        print('dump_images -->')
        # print out images every so often
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
        count_model_params(self.enc_vars, 'AlexNet')
        count_model_params(self.cls_vars, 'Classifier')
        count_model_params(t_vars, 'Total')


def to_string2(li, elem_sep=","):
    st = ''
    for e in li:
        st += e.decode("utf-8")
        if elem_sep:
            st += elem_sep
    st = st[:-1]
    return st


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


