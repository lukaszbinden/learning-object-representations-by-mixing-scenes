import sys
import tensorflow as tf
import glob
import sys
import numpy as np
from ops_alex import *
from utils_common import *
# from imageio import imsave
from scipy.misc import imsave
import traceback

epochs = 1
batch_size = 2  # must divide dataset size (some strange error occurs if not)
image_size = 224

def main(_):
    with tf.Session() as sess:

        tf.set_random_seed(4285)

        basedir = 'datasets/coco/2017_training/version/v2'
        tfrecords_dir = os.path.join(basedir, 'tmp/')
        file_out_dir = os.path.join(basedir, 'tiles/')

        reader = tf.TFRecordReader()
        read_fn = lambda name : read_record(name, reader, image_size)
        filenames, train_images = get_pipeline(tfrecords_dir, batch_size, epochs, read_fn)

        tile_size = image_size / 2
        assert tile_size.is_integer()
        tile_size = int(tile_size)

        # create tiles for I1
        tile1 = tf.image.crop_to_bounding_box(train_images, 0, 0, tile_size, tile_size)
        tile1 = resize(tile1, image_size)
        tile2 = tf.image.crop_to_bounding_box(train_images, 0, tile_size, tile_size, tile_size)
        tile2 = resize(tile2, image_size)
        tile3 = tf.image.crop_to_bounding_box(train_images, tile_size, 0, tile_size, tile_size)
        tile3 = resize(tile3, image_size)
        tile4 = tf.image.crop_to_bounding_box(train_images, tile_size, tile_size, tile_size, tile_size)
        tile4 = resize(tile4, image_size)

        ########################################################################################################

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        try:
            cnt = 0
            while not coord.should_stop():

                tr_imgs, t1, t2, t3, t4, fns = sess.run([train_images, tile1, tile2, tile3, tile4, filenames])

                for i in range(t1.shape[0]):
                    filename = fns[i].decode("utf-8")
                    tr_img = tr_imgs[i]
                    t1_s = t1[i]
                    t2_s = t2[i]
                    t3_s = t3[i]
                    t4_s = t4[i]

                    filedir_t = os.path.join(basedir, 'full')
                    t_name = os.path.join(filedir_t, filename)
                    imsave(t_name, tr_img)

                    filedir_t = os.path.join(file_out_dir, 't1')
                    fn_base = filename.split('.jpg')[0]
                    t_name = os.path.join(filedir_t, fn_base + '_t1.jpg')
                    imsave(t_name, t1_s)
                    filedir_t = os.path.join(file_out_dir, 't2')
                    t_name = os.path.join(filedir_t, fn_base + '_t2.jpg')
                    imsave(t_name, t2_s)
                    filedir_t = os.path.join(file_out_dir, 't3')
                    t_name = os.path.join(filedir_t, fn_base + '_t3.jpg')
                    imsave(t_name, t3_s)
                    filedir_t = os.path.join(file_out_dir, 't4')
                    t_name = os.path.join(filedir_t, fn_base + '_t4.jpg')
                    imsave(t_name, t4_s)

                    cnt = cnt + 1
                    if cnt % 300 == 0:
                    	print(cnt)


        except Exception as e:
            if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Exception here, ending training.. cnt = %d' % cnt)
                if filename:
                    print('filename: %s' % filename)
                tb = traceback.format_exc()
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(e)
                print(tb)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

        print('done.')


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
                                         enqueue_many=False,
                                         allow_smaller_final_batch=False)

def resize(image, img_size):
    image = tf.image.resize_images(image, [img_size, img_size])
    if len(image.shape) == 4: # check for batch case
        image = tf.reshape(image, (batch_size, img_size, img_size, 3))
        with tf.control_dependencies([tf.assert_equal(batch_size, image.shape[0])]):
            return image
    else:
        return tf.reshape(image, (img_size, img_size, 3))


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

    oi1 = tf.image.decode_jpeg(orig_image, channels=0)
    size = tf.minimum(img_h, img_w)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape, seed=4285)
    image = resize(image, img_size)
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return filename, image


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)


