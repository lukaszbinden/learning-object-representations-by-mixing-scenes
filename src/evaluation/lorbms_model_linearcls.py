import signal
from utils_dcgan import *
from utils_common import *
from input_pipeline import *
from autoencoder_dblocks import encoder_dense
from constants import *
import numpy as np
from scipy.misc import imsave
import traceback
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import collections

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

        # isIdeRun = 'lz826' in os.path.realpath(sys.argv[0])
        # file_train = self.params.tfrecords_path if not isIdeRun else 'data/train-00011-of-00060.tfrecords'

        ####################################################################################
        # reader = tf.TFRecordReader()
        # rrm_fn = lambda name : read_record_max(name, reader, image_size)
        # filenames, train_images, t1_10nn_ids, t1_10nn_subids, t1_10nn_L2, t2_10nn_ids, t2_10nn_subids, t2_10nn_L2, t3_10nn_ids, t3_10nn_subids, t3_10nn_L2, t4_10nn_ids, t4_10nn_subids, t4_10nn_L2 = \
        #         get_pipeline(file_train, self.batch_size, self.epochs, rrm_fn)
        # print('train_images.shape..:', train_images.shape)


        ########################### STL-10 BEGIN

        # Reads an image from a file, decodes it into a dense tensor, and resizes it
        # to a fixed shape.
        def _parse_function(file, label):
            image_resized = tf.image.resize_images(file, [64, 64])
            # image = tf.reshape(image, (image_size, image_size, 3))
            image_resized = tf.cast(image_resized, tf.float32) * (2. / 255) - 1
            return image_resized, label

        self.images_plh = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
        self.labels_plh = tf.placeholder(tf.int32, shape=[None])
        dataset = tf.data.Dataset.from_tensor_slices((self.images_plh, self.labels_plh)).repeat().shuffle(self.batch_size).batch(self.batch_size)
        self.dataset = dataset.map(_parse_function)

        self.iterator = self.dataset.make_initializable_iterator()
        images, labels = self.iterator.get_next() # Notice: for both train + test images!!
        images = tf.reshape(images, [self.batch_size, 64, 64, 3])
        print("images: ", images)
        print("labels: ", labels)
        y = tf.one_hot(labels, 10, dtype=tf.int32)
        y_onehot = tf.reshape(y, [self.batch_size, 10])
        ########################### STL-10 END

        self.images_I_ref = images
        self.labels = labels
        self.labels_onehot = y_onehot

        with tf.variable_scope('generator'):
            model = self.params.autoencoder_model
            coordConvLayer = True
            ####################

            self.I_ref_f = encoder_dense(self.images_I_ref, self.batch_size, self.feature_size, dropout_p=0.0, preset_model=model, addCoordConv=coordConvLayer)
            print("self.I_ref_f: ", self.I_ref_f.shape)


        with tf.variable_scope('classifier'):
            self.lin_cls_logits = self.linear_classifier(self.I_ref_f)


        with tf.variable_scope('classifier_loss'):
            #self.cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.lin_cls_logits, labels=tf.cast(self.labels, tf.float32)))
            self.cls_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels_onehot, logits=self.lin_cls_logits, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

            _, self.acc_update_op = tf.metrics.accuracy(labels=tf.argmax(self.labels_onehot, axis=1), predictions=tf.argmax(self.lin_cls_logits, axis=1, output_type=tf.int32))

        t_vars = tf.trainable_variables()
        self.gen_vars = [var for var in t_vars if 'generator' in var.name and 'g_' in var.name] # encoder + decoder (generator)
        self.cls_vars = [var for var in t_vars if 'classifier' in var.name]

        list = []
        list.extend(self.gen_vars)
        list.extend(self.cls_vars)
        assert collections.Counter(list) == collections.Counter(t_vars)
        del list

        # only save CLS (not encoder)
        self.saver = tf.train.Saver(self.cls_vars, max_to_keep=5)
        if self.params.is_train:
            self.saver_metrics = tf.train.Saver(self.cls_vars, max_to_keep=None)
        print("build_model() ------------------------------------------<")
        # END of build_model


    def train(self, params):
        """Train DCGAN"""

        if params.continue_from_iteration:
            iteration = params.continue_from_iteration
        else:
            iteration = 0

        global_step = tf.Variable(iteration, name='global_step', trainable=False)

        cls_learning_rate = tf.constant(params.learning_rate_cls)
        # cls_learning_rate = tf.train.exponential_decay(cls_learning_rate, global_step=global_step,
        #                                                   decay_steps=800, decay_rate=0.9, staircase=True)
        print('cls_learning_rate: %s' % cls_learning_rate)

        t_vars = tf.trainable_variables()
        # restore encoder from checkpoint
        print("params.encoder_type: %s" % params.encoder_type)
        if params.encoder_type == "lorbms_frozen":
            self.restore_encoder(params)
        elif params.encoder_type == "lorbms_finetune":
            self.restore_encoder(params)
            self.gen_vars = []
            self.cls_vars = t_vars  # use all vars incl. encoder for training (finetuning)
        elif params.encoder_type == "stl-10":
            self.gen_vars = []
            self.cls_vars = t_vars # use all vars incl. encoder for training
        elif params.encoder_type == "coco":
            assert 1 == 0, "tbi"
        else:
            assert params.encoder_type == "random"

        self.print_model_params(t_vars)

        # for classifier
        c_optim = tf.train.AdamOptimizer(learning_rate=cls_learning_rate, beta1=0.5) \
                          .minimize(self.cls_loss, var_list=self.cls_vars, global_step=global_step)  # params.beta1

        tf.global_variables_initializer().run()

        if params.continue_from:
            ckpt_name = self.load(params, params.continue_from_iteration)
            iteration = int(ckpt_name[ckpt_name.rfind('-')+1:])
            print('continuing from \'%s\'...' % ckpt_name)
            global_step.load(iteration) # load new initial value into variable


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

        self.sess.run(self.iterator.initializer, feed_dict={self.images_plh: X_train_raw, self.labels_plh: y_train})

        n_batches = X_train_raw.shape[0] // self.batch_size + 1
        print("n_batches: %s, X_train_raw.shape[0]: %s, self.batch_size: %s" % (str(n_batches), str(X_train_raw.shape[0]), str(self.batch_size)))

        sum_train_loss_results = []
        sum_train_accuracy_results = []
        for i in range(params.epochs):
            train_loss_results = []
            train_accuracy_results = []
            for _ in range(n_batches):
                _, loss_value, acc_value, gl, lr = self.sess.run([c_optim, self.cls_loss, self.acc_update_op, global_step, cls_learning_rate])
                train_loss_results.append(loss_value)
                sum_train_loss_results.append(loss_value)
                train_accuracy_results.append(acc_value)
                sum_train_accuracy_results.append(acc_value)
            print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}, Global step: {}, LR: {}".format(i + 1, np.mean(train_loss_results), np.mean(train_accuracy_results), str(gl), str(lr)))


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
        self.sess.run(self.iterator.initializer, feed_dict={self.images_plh: X_test_raw, self.labels_plh: y_test})
        test_loss_results = []
        test_accuracy_results = []

        n_batches = X_test_raw.shape[0] // self.batch_size + 1
        print("n_batches: %s, X_test_raw.shape[0]: %s, self.batch_size: %s" % (str(n_batches), str(X_test_raw.shape[0]), str(self.batch_size)))

        for _ in range(n_batches):
            loss_value, acc_value = self.sess.run([self.cls_loss, self.acc_update_op])
            test_loss_results.append(loss_value)
            test_accuracy_results.append(acc_value)

        #print("Test losses: ", test_loss_results)
        #print("Test accuracies:", test_accuracy_results)
        print("Test loss: avg: %f, std: %f" % (np.mean(test_loss_results), np.std(test_loss_results)))
        print("Test acc.: avg: %f, std: %f" % (np.mean(test_accuracy_results), np.std(test_accuracy_results)))

        # END of train()


    def test(self, params):
        """Test DCGAN"""
        """For each image in the test set create a mixed scene and save it (ie run for 1 epoch)."""

        tf.global_variables_initializer().run()

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

        # simple mechanism to coordinate the termination of a set of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            signal.signal(signal.SIGTERM, self.handle_exit)

            num_gen_imgs = 0

            file_out_dir = params.metric_fid_out_dir

            while not coord.should_stop():
                images_mix = self.sess.run(self.images_I_ref_I_M_mix)

                for i in range(self.batch_size): # for each image in batch
                    num_gen_imgs = num_gen_imgs + 1
                    img_mix = images_mix[i]

                    t_name = os.path.join(file_out_dir, 'img_mix_gen_%s.jpg' % num_gen_imgs)
                    imsave(t_name, img_mix)

                    if num_gen_imgs % 300 == 0:
                    	print(num_gen_imgs)

                if self.end:
                    print('going to shutdown now...')
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
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

        # END of test()


    def path(self, filename):
        return os.path.join(self.params.summary_dir, filename)

    def linear_classifier(self, features):
        features = tf.reshape(features, [self.batch_size, -1])
        logits = tf.layers.dense(inputs=features, units=self.params.number_of_classes, use_bias=True, activation=None,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02, seed=4285),
                                 bias_initializer=tf.constant_initializer(0.02),
                                 name='Linear')
        return logits

    def restore_encoder(self, params):
        enc_vars = [var.name for var in self.gen_vars if 'g_1' in var.name]
        variables = slim.get_variables_to_restore(include=enc_vars)
        # print("variables1: ", variables)

        path = params.encoder_checkpoint_name if not self.isIdeRun else "../exp70/checkpoint/DCGAN.model-50"
        print('restoring encoder to [%s]...' % path)
        init_restore_op, init_feed_dict  = slim.assign_from_checkpoint(model_path=path, var_list=variables)
        self.sess.run(init_restore_op, feed_dict=init_feed_dict)
        print('encoder restored.')


    def make_summary_ops(self):
        # tf.summary.scalar('loss_g', self.g_loss)
        # tf.summary.scalar('loss_g_comp', g_loss_comp)
        # tf.summary.scalar('loss_L2', losses_l2)
        tf.summary.scalar('loss_cls', self.cls_loss)
        # tf.summary.scalar('loss_dsc', self.dsc_loss)
        # tf.summary.scalar('loss_dsc_fake', self.dsc_loss_fake)
        # tf.summary.scalar('loss_dsc_real', self.dsc_loss_real)
        # tf.summary.scalar('rec_loss_Iref_hat_I_ref', self.rec_loss_I_ref_hat_I_ref)
        # tf.summary.scalar('rec_loss_I_ref_4_I_ref', self.rec_loss_I_ref_4_I_ref)
        # tf.summary.scalar('rec_loss_I_t1_hat_I_t1', self.rec_loss_I_t1_hat_I_t1)
        # tf.summary.scalar('rec_loss_I_t1_4_I_t1', self.rec_loss_I_t1_4_I_t1)
        # tf.summary.scalar('rec_loss_I_t2_4_I_t2', self.rec_loss_I_t2_4_I_t2)
        # tf.summary.scalar('rec_loss_I_t3_4_I_t3', self.rec_loss_I_t3_4_I_t3)
        # tf.summary.scalar('rec_loss_I_t4_4_I_t4', self.rec_loss_I_t4_4_I_t4)
        # tf.summary.scalar('psnr_images_I_ref_hat', self.images_I_ref_hat_psnr)
        # tf.summary.scalar('psnr_images_I_ref_4', self.images_I_ref_4_psnr)
        # tf.summary.scalar('psnr_images_t1_4', self.images_t1_4_psnr)
        # tf.summary.scalar('psnr_images_t3_4', self.images_t3_4_psnr)
        # tf.summary.scalar('dsc_I_ref_mean', self.dsc_I_ref_mean)
        # tf.summary.scalar('dsc_I_ref_I_M_mix_mean', self.dsc_I_ref_I_M_mix_mean)
        # tf.summary.scalar('V_G_D', self.v_g_d)
        # tf.summary.scalar('c_learning_rate', self.c_learning_rate)
        #
        # images = tf.concat(
        #     tf.split(tf.concat([self.images_I_ref, self.images_I_ref_hat, self.images_I_ref_4,
        #            self.images_I_M_mix, self.images_I_ref_I_M_mix], axis=2), self.batch_size,
        #              axis=0), axis=1)
        # tf.summary.image('images', images)
        #
        # #_ TODO add actual test images/mixes later
        # #_ tf.summary.image('images_I_test_hat', self.images_I_test_hat)
        #
        # accuracy1 = tf.metrics.accuracy(predictions=tf.argmax(self.assignments_predicted_t1, 1),
        #                                       labels=tf.argmax(self.assignments_actual_t1, 1),
        #                                       updates_collections=tf.GraphKeys.UPDATE_OPS)
        # tf.summary.scalar('classifier/accuracy_t1_result', accuracy1[1])
        # accuracy2 = tf.metrics.accuracy(predictions=tf.argmax(self.assignments_predicted_t2, 1),
        #                                labels=tf.argmax(self.assignments_actual_t2, 1),
        #                                updates_collections=tf.GraphKeys.UPDATE_OPS)
        # tf.summary.scalar('classifier/accuracy_t2_result', accuracy2[1])
        # accuracy3 = tf.metrics.accuracy(predictions=tf.argmax(self.assignments_predicted_t3, 1),
        #                                labels=tf.argmax(self.assignments_actual_t3, 1),
        #                                updates_collections=tf.GraphKeys.UPDATE_OPS)
        # tf.summary.scalar('classifier/accuracy_t3_result', accuracy3[1])
        # accuracy4 = tf.metrics.accuracy(predictions=tf.argmax(self.assignments_predicted_t4, 1),
        #                                labels=tf.argmax(self.assignments_actual_t4, 1),
        #                                updates_collections=tf.GraphKeys.UPDATE_OPS)
        # tf.summary.scalar('classifier/accuracy_t4_result', accuracy4[1])
        # return accuracy1[1], accuracy2[1], accuracy3[1], accuracy4[1]


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

        # grid_size = np.ceil(np.sqrt(self.batch_size))
        # grid = [grid_size, grid_size]
        # save_images(images_Iref, grid, self.path('%s_images_I_ref.jpg' % counter))
        # save_images(images_IM, grid, self.path('%s_images_I_M.jpg' % counter))
        # st = ''
        # for list in ass_actual:
        #     for e in list:
        #         st += str(e)
        #     st += '_'
        # st = st[:-1]
        # file_path = self.path('%s_images_Iref_IM_mix_IN_%s.jpg' % (counter, st))
        # save_images(images_IrefImMixIn, grid, file_path)
        # file_path = self.path('%s_images_Iref_IM_mix_OUT_%s.jpg' % (counter, st))
        # save_images(images_IrefImMixOut, grid, file_path)
        # save_images(images_Iref4, grid, self.path('%s_images_I_ref_4.jpg' % counter))
        # save_images(images_IM5, grid, self.path('%s_images_I_M_5.jpg' % counter))

        st = to_string(ass_actual)
        act_batch_size = min(self.batch_size, 16)

        # grid = [act_batch_size, 3]
        # save_images_multi(img_I_ref, img_I_M_mix, img_I_ref_I_M_mix, grid, act_batch_size, self.path('%s_images_I_ref_I_M_mix_%s.jpg' % (counter, st)), maxImg=act_batch_size)

        grid = [act_batch_size, 5]
        save_images_5cols(img_I_ref, img_I_ref_hat, img_I_ref_4, img_I_M_mix, img_I_ref_I_M_mix, grid, act_batch_size, self.path('%s_images_I_ref_I_M_mix_%s.jpg' % (counter, st)), maxImg=act_batch_size)

        print("filenames iteration %d: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" % counter)
        print("filenames I_ref..: %s" % to_string2(fnames_Iref))
        print("filenames I_t1...: %s" % to_string2(fnames_t1))
        print("filenames I_t2...: %s" % to_string2(fnames_t2))
        print("filenames I_t3...: %s" % to_string2(fnames_t3))
        print("filenames I_t4...: %s" % to_string2(fnames_t4))
        print("filenames iteration %d: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" % counter)

        # grid = [act_batch_size, 1]
        # save_images(img_I_ref_hat, grid, self.path('%s_I_ref_hat.jpg' % counter), maxImg=act_batch_size)
        # save_images(img_I_ref_4, grid, self.path('%s_I_ref_4.jpg' % counter), maxImg=act_batch_size)
        # save_images(img_t1, grid, self.path('%s_images_t1.jpg' % counter), maxImg=act_batch_size)
        # save_images(img_t2, grid, self.path('%s_images_t2.jpg' % counter), maxImg=act_batch_size)
        # save_images(img_t2_4, grid, self.path('%s_images_t2_4.jpg' % counter), maxImg=act_batch_size)
        # save_images(img_t3, grid, self.path('%s_images_t3.jpg' % counter), maxImg=act_batch_size)
        # save_images(img_t4, grid, self.path('%s_images_t4.jpg' % counter), maxImg=act_batch_size)

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


