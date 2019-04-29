from utils_common import *
from input_pipeline import *
from constants import *
import collections
import traceback
import tensorflow.contrib.slim as slim
import numpy as np
from alexnet import alexnet_v2
# import scipy.misc

LEARNING_RATE = 1e-03


# LZ 27.04:
# This training and test program is inspired by
# https://github.com/dontfollowmeimcrazy/imagenet/

class DCGAN(object):

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

        self.build_model()


    def build_model(self):
        print("build_model() ------------------------------------------>")
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        tf.set_random_seed(self.random_seed)

        # image_size = self.image_size
        self.feature_size_tile = self.params.chunk_size * self.params.chunk_num
        self.feature_size = self.feature_size_tile * NUM_TILES_L2_MIX


        ########################### ImageNet BEGIN

        image_size = self.image_size

        isIdeRun = 'lz826' in os.path.realpath(sys.argv[0])
        file_train = self.params.dataset_path if not isIdeRun else '../data/imagenet_00122-of-00128.tfrecords' # imagenet_00102-of-00128.tfrecords

        reader = tf.TFRecordReader()
        rrm_fn = lambda name : read_record(name, reader, image_size)
        train_images, labels = get_pipeline(file_train, self.batch_size, self.epochs, rrm_fn)
        y_onehot = tf.one_hot(labels, 1000, dtype=tf.int32)
        y_onehot = tf.reshape(y_onehot, [self.batch_size, 1000])
        ########################### ImageNet END

        self.images_I_ref = train_images
        self.labels_onehot = y_onehot

        self.isTrainingAlexnetPlh = tf.placeholder(tf.bool)

        with tf.variable_scope('alexnet'):
            self.lin_cls_logits, _ = self.alexnet(self.images_I_ref, is_training=self.isTrainingAlexnetPlh)

        with tf.variable_scope('classifier_loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_onehot, logits=self.lin_cls_logits, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('l2_loss'):
            LAMBDA = 5e-04  # for weight decay
            lmbda = LAMBDA

            def get_weights():
                return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.endswith('weights:0')]

            l2_loss = tf.reduce_sum(lmbda * tf.stack([tf.nn.l2_loss(v) for v in get_weights()]))
            tf.summary.scalar('l2_loss', l2_loss)

        with tf.name_scope('loss'):
            self.cls_loss = cross_entropy + l2_loss
            tf.summary.scalar('cls_loss', self.cls_loss)

        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(self.lin_cls_logits, 1), tf.argmax(self.labels_onehot, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        t_vars = tf.trainable_variables()
        self.an_vars = [var for var in t_vars if 'alexnet' in var.name]

        self.print_model_params(t_vars)

        list = []
        list.extend(self.an_vars)
        assert collections.Counter(list) == collections.Counter(t_vars)
        del list

        # only save encoder
        self.saver = tf.train.Saver(self.an_vars, max_to_keep=5)
        print("build_model() ------------------------------------------<")
        # END of build_model


    def train(self, params):
        """Train DCGAN"""

        if params.continue_from_iteration:
            iteration = params.continue_from_iteration
        else:
            iteration = 0

        global_step = tf.Variable(iteration, name='global_step', trainable=False)

        # self.cls_learning_rate = tf.train.exponential_decay(learning_rate=params.learning_rate_cls, global_step=global_step, decay_steps=10000, decay_rate=0.5, staircase=True)
        # print('cls_learning_rate: %s' % self.cls_learning_rate)
        lr = tf.placeholder(tf.float32)

        c_optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9) \
                          .minimize(self.cls_loss, var_list=self.an_vars, global_step=global_step)

        self.initialize_uninitialized(tf.global_variables(), "global")
        self.initialize_uninitialized(tf.local_variables(), "local")

        if params.continue_from:
            self.restore_alexnet(params.continue_from_checkpoint_name)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        #self.make_summary_ops()

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(params.summary_dir)
        summary_writer.add_graph(self.sess.graph)

        learning_rate = LEARNING_RATE

        try:
            iter_per_epoch = (self.params.num_images / self.batch_size)

            # Training
            while not coord.should_stop():
                summary_str, _, step = self.sess.run([summary_op, c_optim, global_step], feed_dict={lr: learning_rate, self.isTrainingAlexnetPlh: True})
                summary_writer.add_summary(summary_str, step)

                # decaying learning rate
                if step == 170000 or step == 350000:
                    learning_rate /= 10

                iteration += 1

                epoch = int(iteration // iter_per_epoch) + 1

                # display current training information
                if step % 100 == 0:
                    c, a = self.sess.run([self.cls_loss, self.accuracy], feed_dict={lr: learning_rate, self.isTrainingAlexnetPlh: False})
                    print('Epoch: {:02d} Step/Batch: {:07d} Iteration: {:07d} --- Loss: {:.5f} Training accuracy: {:.4f}'.format(epoch, step, iteration, c, a))

                if epoch >= self.epochs:
                    print("epoch limit reached, terminating...")
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
        finally:
            if iteration > 0:
                self.save(params.checkpoint_dir, iteration)  # save model again
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)
        # END of train()


    def test(self, params):
        """Test DCGAN"""
        """For each image in the test set create a mixed scene and save it (ie run for 1 epoch)."""

        print("test -->")

        self.restore_alexnet(params.alexnet_checkpoint_name)

        self.initialize_uninitialized(tf.global_variables(), "global")
        self.initialize_uninitialized(tf.local_variables(), "local")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        iteration = 0
        test_accuracy_results = []

        try:
            # Test
            while not coord.should_stop():
                iteration += 1
                c, a = self.sess.run([self.cls_loss, self.accuracy], feed_dict={self.isTrainingAlexnetPlh: False})
                print('Iteration: {:07d} --- Loss: {:.5f} Test accuracy: {:.4f}'.format(iteration, c, a))
                test_accuracy_results.append(a)

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

            test_accuracy = np.mean(test_accuracy_results)
            test_std = np.std(test_accuracy_results)
            print("Accuracy on validation set (Top-1): avg: %f, std: %f" % (test_accuracy, test_std))

            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


        # END of test()
        print("test <--")


    def restore_alexnet(self, chkp_name):
        al_var_names = [var.name for var in self.an_vars]
        variables = slim.get_variables_to_restore(include=al_var_names)
        # print("variables1: ", variables)

        path = chkp_name if not self.isIdeRun else "../checkpoints/exp70/checkpoint/DCGAN.model-50"
        print('restoring alexnet from [%s]...' % path)
        init_restore_op, init_feed_dict  = slim.assign_from_checkpoint(model_path=path, var_list=variables)
        self.sess.run(init_restore_op, feed_dict=init_feed_dict)
        print('alexnet restored.')


    def initialize_uninitialized(self, vars, context):
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in vars])
        not_initialized_vars = [v for (v, f) in zip(vars, is_not_initialized) if not f]

        print("#not initialized variables '%s': %d" % (context, len(not_initialized_vars))) # only for testing
        if len(not_initialized_vars):
            print("initializing not_initialized_vars: %s" % str(not_initialized_vars))
            self.sess.run(tf.variables_initializer(not_initialized_vars))


    def path(self, filename):
        return os.path.join(self.params.summary_dir, filename)


    # multi-label classifier
    def linear_classifier(self, features):
        features = tf.reshape(features, [self.batch_size, -1])
        print("features: ", features.shape)
        logits = tf.layers.dense(inputs=features, units=self.params.number_of_classes, use_bias=True, activation=None,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02, seed=4285),
                                 bias_initializer=tf.constant_initializer(0.02),
                                 name='Linear')
        return logits

    def alexnet(self, images, is_training):
        return alexnet_v2(images, is_training=is_training)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        path = os.path.join(checkpoint_dir, self.model_name)
        get_pp().pprint('[1] Save model to {} with step={}'.format(path, step))
        self.saver.save(self.sess, path, global_step=step)
        # Save model after every epoch -> is more coherent than iterations
        # if step > 1 and np.mod(step, 25000) == 0:
        #    self.save_metrics(step)

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

    def print_model_params(self, t_vars):
        count_model_params(self.an_vars, 'AlexNet')
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


def read_record(filename_queue, reader, image_size, crop=True):
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)
    class_id = features['image/class/label']
    orig_image = features['image/encoded']

    oi1 = tf.image.decode_jpeg(orig_image, channels=3)

    if crop:
        # LZ: scale image to 256px on smaller side, then random crop at 224x224
        oi1 = tf.cond(tf.less(img_h, img_w),
                true_fn=lambda: resize_scale_w(oi1, img_h, img_w),
                false_fn=lambda: resize_scale_h(oi1, img_h, img_w))
        crop_shape = [224, 224, 3]
        image = tf.random_crop(oi1, crop_shape, seed=4285)
    else:
        assert 1 == 0, "notsupported"
    image = tf.reshape(image, (image_size, image_size, 3), name="final_reshape")
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return image, class_id


def resize_scale_w(imag, ih, iw):
    # w = int(float(256 * iw) / ih)
    r = tf.cast(256 * iw, tf.float32)
    ihf = tf.cast(ih, tf.float32)
    w = tf.cast(tf.div(r, ihf), tf.int32)
    shape = tf.parallel_stack([256, w, 3])
    imag = tf.expand_dims(imag, 0)
    imag = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
                    true_fn=lambda: tf.image.resize_bilinear(imag, shape[:2], align_corners=False),
                    false_fn=lambda: tf.image.resize_bicubic(imag, shape[:2], align_corners=False))
    imag = tf.squeeze(imag)
    return tf.reshape(imag, (256, w, 3), name="resize_scale_w_reshape")


def resize_scale_h(imag, ih, iw):
    # h = int(float(256 * ih) / iw)
    r = tf.cast(256 * ih, tf.float32)
    iwf = tf.cast(iw, tf.float32)
    h = tf.cast(tf.div(r, iwf), tf.int32)
    shape = tf.parallel_stack([h, 256, 3])
    imag = tf.expand_dims(imag, 0)
    imag = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
                    true_fn=lambda: tf.image.resize_bilinear(imag, shape[:2], align_corners=False),
                    false_fn=lambda: tf.image.resize_bicubic(imag, shape[:2], align_corners=False))
    imag = tf.squeeze(imag)
    return tf.reshape(imag, (h, 256, 3), name="resize_scale_h_reshape")
