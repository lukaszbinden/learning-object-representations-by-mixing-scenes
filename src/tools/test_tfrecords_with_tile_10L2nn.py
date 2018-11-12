import tensorflow as tf
import glob
import sys
import os
import pickle
from scipy.misc import imsave
import traceback
from create_tfrecords_with_tile_10L2nn import ImageCoder

def main(_):
    with tf.Session() as sess:

        tf.set_random_seed(4285)

        epochs = 1
        batch_size = 4 # must divide dataset size (some strange error occurs if not)
        image_size = 128

        tfrecords_file = '/data/cvg/lukas/datasets/coco/2017_training/tfrecords_l2mix_flip_tile_10-L2nn_4285/'
        reader = tf.TFRecordReader()
        read_fn = lambda name : read_record(name, reader, image_size)
        filename, train_images, t1_10nn_str, t2_10nn_str, t3_10nn_str, t4_10nn_str, \
                t1_10nn_strs, t2_10nn_strs, t3_10nn_strs, t4_10nn_strs = \
                get_pipeline(tfrecords_file, batch_size, epochs, read_fn)

        t1_10nn_str = tf.reshape(tf.sparse.to_dense(t1_10nn_str), (batch_size, 10))
        t2_10nn_str = tf.reshape(tf.sparse.to_dense(t2_10nn_str), (batch_size, 10))
        t3_10nn_str = tf.reshape(tf.sparse.to_dense(t3_10nn_str), (batch_size, 10))
        t4_10nn_str = tf.reshape(tf.sparse.to_dense(t4_10nn_str), (batch_size, 10))

        t1_10nn_strs = tf.reshape(tf.sparse.to_dense(t1_10nn_strs), (batch_size, 10))
        t2_10nn_strs = tf.reshape(tf.sparse.to_dense(t2_10nn_strs), (batch_size, 10))
        t3_10nn_strs = tf.reshape(tf.sparse.to_dense(t3_10nn_strs), (batch_size, 10))
        t4_10nn_strs = tf.reshape(tf.sparse.to_dense(t4_10nn_strs), (batch_size, 10))

        # [('000000000927_1.jpg', 0.03125), ('000000568135_2.jpg', 19095.953), ('000000187857_1.jpg', 23359.39),
        #  ('000000521998_2.jpg', 23557.688), ('000000140816_1.jpg', 24226.852), ('000000015109_1.jpg', 25191.469),
        #  ('000000525567_1.jpg', 25484.93), ('000000377422_1.jpg', 25654.125), ('000000269815_2.jpg', 26794.836),
        #  ('000000345617_2.jpg', 26872.812)]

        ########################################################################################################

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        filedir = '../logs/test/test_tfrecords_with_tile_10L2nn'

        max = 5
        cnt = 0
        coder = ImageCoder()

        try:
            cnt = 0
            while not coord.should_stop():
                print('t1_10nn_str type: %s' % type(t1_10nn_str))
                fns, ti, t110nn, t210nn, t310nn, t410nn, t1s, t2s, t3s, t4s = sess.run([filename, train_images, t1_10nn_str, t2_10nn_str, t3_10nn_str, t4_10nn_str,
                                                                                    t1_10nn_strs, t2_10nn_strs,
                                                                                    t3_10nn_strs, t4_10nn_strs])
                print('ti.shape: %s' % str(ti.shape[0]))
                print('t110nn type: %s' % type(t110nn))
                print('t110nn str: %s' % str(t110nn))
                print('t1s str: %s' % str(t1s))

                for i in range(ti.shape[0]):
                    print('ITERATION [%d]' % i)
                    fname = fns[i].decode("utf-8")
                    t1_knn = t110nn[i]
                    t1_knns = t1s[i]
                    t2_knn = t210nn[i]
                    t2_knns = t1s[i]
                    t3_knn = t310nn[i]
                    t3_knns = t1s[i]
                    t4_knn = t410nn[i]
                    t4_knns = t1s[i]

                    print('>>>>>>>>>>>>>>>>>>>>')
                    print('file: %s' % fname)
                    # print('t1_10nn type: %s' % str(type(t1_knn)))
                    print('t1_10nn shape: %s' % str(t1_knn.shape))
                    print('t1_10nn: %s' % str(t1_knn))
                    #print('t2_10nn: %s' % str(t2_knn))
                    #print('t3_10nn: %s' % str(t3_knn))
                    #print('t4_10nn: %s' % str(t4_knn))
                    print('t1_knns shape: %s' % str(t1_knns.shape))
                    print('t1_knns: %s' % str(t1_knns))
                    #print('t2_knns: %s' % str(t2_knns))
                    #print('t3_knns: %s' % str(t3_knns))
                    #print('t4_knns: %s' % str(t4_knns))
                    print('<<<<<<<<<<<<<<<<<<<<')
                    print('')
                    print('')

                    name = os.path.join(filedir, fname)
                    imsave(name, ti[i])


                cnt = cnt + 1
                if cnt >= max:
                    break


        except Exception as e:
            if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Exception here, ending training..')
                tb = traceback.format_exc()
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(e)
                print(tb)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


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
                'image/knn/t1': tf.VarLenFeature(tf.int64),
                'image/knn/t1s': tf.VarLenFeature(tf.int64),
                'image/knn/t2': tf.VarLenFeature(tf.int64),
                'image/knn/t2s': tf.VarLenFeature(tf.int64),
                'image/knn/t3': tf.VarLenFeature(tf.int64),
                'image/knn/t3s': tf.VarLenFeature(tf.int64),
                'image/knn/t4': tf.VarLenFeature(tf.int64),
                'image/knn/t4s': tf.VarLenFeature(tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)
    filename = features['image/filename']

    t1_10nn_str = features['image/knn/t1']
    t1_10nn_strs = features['image/knn/t1s']
    t2_10nn_str = features['image/knn/t2']
    t2_10nn_strs = features['image/knn/t2s']
    t3_10nn_str = features['image/knn/t3']
    t3_10nn_strs = features['image/knn/t3s']
    t4_10nn_str = features['image/knn/t4']
    t4_10nn_strs = features['image/knn/t4s']

    orig_image = features['image/encoded']

    image = preprocess_image(orig_image, img_size, img_w, img_h)

    return filename, image, t1_10nn_str, t2_10nn_str, t3_10nn_str, t4_10nn_str, t1_10nn_strs, t2_10nn_strs, t3_10nn_strs, t4_10nn_strs


def preprocess_image(orig_image, img_size, img_w, img_h):
    oi1 = tf.image.decode_jpeg(orig_image)
    size = tf.minimum(img_h, img_w)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape)
    image = tf.image.resize_images(image, [img_size, img_size])
    image = tf.reshape(image, (img_size, img_size, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1
    return image


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)


