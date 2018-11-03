import sys
import tensorflow as tf
import glob
import sys
import numpy as np
import scipy
from ops_alex import *
from utils_common import *
import pickle
from scipy.misc import imsave

def main(_):
    with tf.Session() as sess:

        tf.set_random_seed(4285)

        epochs = 1
        batch_size = 2 # must divide dataset size (some strange error occurs if not)
        image_size = 224

        tfrecords_file = 'datasets/coco/2017_training/tfrecords_l2mix_flip/'
        reader = tf.TFRecordReader()
        read_fn = lambda name : read_record(name, reader, image_size)
        filenames, train_images = get_pipeline(tfrecords_file, batch_size, epochs, read_fn)

        features = encoder(train_images, batch_size)

        ########################################################################################################

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        basedir = 'datasets/coco/2017_training'
        filedir = os.path.join(basedir, 'clustering_224x224')
        name = os.path.join(basedir, 'filename_feature_dict.obj')
        handle = open(name, "wb")
        filename_feature_dict = {}
        try:
            cnt = 0
            while not coord.should_stop():
                ti, fns, fs = sess.run([train_images, filenames, features])

                for i in range(ti.shape[0]):
                    filename = fns[i].decode("utf-8")
                    name = os.path.join(filedir, filename)
                    imsave(name, ti[i])
                    filename_feature_dict[filename] = fs[i]
                    cnt = cnt + 1
                    if cnt % 300 == 0:
                    	print(cnt)


        except Exception as e:
            if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Exception here, ending training..')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(e)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        finally:
            print('dict length: %d (should equal dataset size)' % (len(filename_feature_dict)))
            pickle.dump(filename_feature_dict, handle)
            handle.close()
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def encoder(tile_image, batch_size):
        """
        returns: 1D vector f1 with size=self.feature_size
        """
        df_dim = 64

        s0 = lrelu(instance_norm(conv2d(tile_image, df_dim, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, df_dim * 2, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, df_dim * 4, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv2')))
        s3 = lrelu(instance_norm(conv2d(s2, df_dim * 8, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv3')))
        s4 = lrelu(instance_norm(conv2d(s3, df_dim * 8, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv4')))
        rep = lrelu((linear(tf.reshape(s4, [batch_size, -1]), 512, 'g_1_fc')))

        return rep


def get_pipeline(dump_file, batch_size, epochs, read_fn, read_threads=4):
    with tf.variable_scope('dump_reader'):
        with tf.device('/cpu:0'):
            all_files = glob.glob(dump_file + '*')
            all_files = all_files if len(all_files) > 0 else [dump_file]
            print('tfrecords: ' + str(all_files))
            filename_queue = tf.train.string_input_producer(all_files, num_epochs=epochs ,shuffle=True)
            #example_list = [read_record(filename_queue) for _ in range(read_threads)]
            example_list = [read_fn(filename_queue) for _ in range(read_threads)]

            return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                         capacity=100 + batch_size * 16,
                                         min_after_dequeue=100,
                                         enqueue_many=False) #,
                                         #allow_smaller_final_batch=True)


def read_record(filename_queue, reader, img_size):
    # reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)
    filename = features['image/filename']

    orig_image = features['image/encoded']

    oi1 = tf.image.decode_jpeg(orig_image)
    size = tf.minimum(img_h, img_w)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape)
    image = tf.image.resize_images(image, [img_size, img_size])
    image = tf.reshape(image, (img_size, img_size, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return filename, image


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)


