import tensorflow as tf
import glob
import sys

def main(_):
    with tf.Session() as sess:

        reader = tf.TFRecordReader()
        read_fn = lambda name : read_record(name, reader)
        h, w, crop_shape, train_images = get_pipeline('2017_train_small_anys.tfrecords', 4, 1, read_fn)
        read_fn_sc = lambda name : read_record_2(name, reader, 9)
        h2, w2, crop_shape2, train_images_2 = get_pipeline('2017_train_small_anys.tfrecords', 4, 1, read_fn_sc)
        read_fn_sc = lambda name : read_record_2(name, reader, 8)
        h3, w3, crop_shape3, train_images_3 = get_pipeline('2017_train_small_anys.tfrecords', 4, 1, read_fn_sc)
        read_fn_sc = lambda name : read_record_2(name, reader, 7)
        h4, w4, crop_shape4, train_images_4 = get_pipeline('2017_train_small_anys.tfrecords', 4, 1, read_fn_sc)
        read_fn_sc = lambda name : read_record_2(name, reader, 6)
        h5, w5, crop_shape5, train_images_5 = get_pipeline('2017_train_small_anys.tfrecords', 4, 1, read_fn_sc)
        read_fn_sc = lambda name : read_record_2(name, reader, 5)
        h6, w6, crop_shape6, train_images_6 = get_pipeline('2017_train_small_anys.tfrecords', 4, 1, read_fn_sc)
        read_fn_sc = lambda name : read_record_2(name, reader, 4)
        h7, w7, crop_shape7, train_images_7 = get_pipeline('2017_train_small_anys.tfrecords', 4, 1, read_fn_sc)
        read_fn_sc = lambda name : read_record_2(name, reader, 3)
        h8, w8, crop_shape8, train_images_8 = get_pipeline('2017_train_small_anys.tfrecords', 4, 1, read_fn_sc)

        # imgs_per_group = int(2/2)
        # group = lambda i: train_images[(i * imgs_per_group):imgs_per_group + (i * imgs_per_group), :, :, :]
        # images_i1 = group(0)
        # images_i2 = group(1)

        # for _ in range(3):
        #     print('images_i1.shape[1]: ' + images_i1.shape[1])
        #     print('images_i1.shape[2]: ' + images_i1.shape[2])
        #     size = tf.minimum(images_i1.shape[1], images_i1.shape[2])
        #     print('i1: %s -> %s' % (images_i1.shape, size.shape))
        #     crop_shape = tf.parallel_stack([size, size, 3])
        #     crop = tf.random_crop(images_i1, crop_shape)
        #     print(type(crop))
        #     print(crop.shape)

        ######################################################################3

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        try:
            while not coord.should_stop():
                #hr, wr, i1, i2 = sess.run([h, w, images_i1, images_i2])
                hr, wr, c, hr2, wr2, c2 \
                    , hr3, wr3, c3 \
                    , hr4, wr4, c4 \
                    , hr5, wr5, c5 \
                    , hr6, wr6, c6 \
                    , hr7, wr7, c7 \
                    , hr8, wr8, c8 = sess.run([h, w, crop_shape \
                                                             , h2, w2, crop_shape2 \
                                                             , h3, w3, crop_shape3 \
                                                             , h4, w4, crop_shape4 \
                                                             , h5, w5, crop_shape5 \
                                                             , h6, w6, crop_shape6 \
                                                             , h7, w7, crop_shape7 \
                                                             , h8, w8, crop_shape8 ])
                print('22222222222222222222222222222222')
                print('hr: %s' % str(hr))
                print('wr: %s' % str(wr))
                print('c:')
                print(c)
                print('22222222222222222222222222222222')
                print('hr2: %s' % str(hr2))
                print('wr2: %s' % str(wr2))
                print('c2:')
                print(c2)
                print('22222222222222222222222222222222')
                print('hr3: %s' % str(hr3))
                print('wr3: %s' % str(wr3))
                print('c3:')
                print(c3)
                print('22222222222222222222222222222222')
                print('hr4: %s' % str(hr4))
                print('wr4: %s' % str(wr4))
                print('c4:')
                print(c4)
                print('22222222222222222222222222222222')
                print('hr5: %s' % str(hr5))
                print('wr5: %s' % str(wr5))
                print('c5:')
                print(c5)
                print('22222222222222222222222222222222')
                print('hr6: %s' % str(hr6))
                print('wr6: %s' % str(wr6))
                print('c6:')
                print(c6)
                print('22222222222222222222222222222222')
                print('hr7: %s' % str(hr7))
                print('wr7: %s' % str(wr7))
                print('c7:')
                print(c7)
                print('22222222222222222222222222222222')
                print('hr8: %s' % str(hr8))
                print('wr8: %s' % str(wr8))
                print('c8:')
                print(c8)
                print('--------------------------------------------')

        except Exception as e:
            if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Exception here, ending training..')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(e)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def get_pipeline(dump_file, batch_size, epochs, read_fn, read_threads=4):
    with tf.variable_scope('dump_reader'):
        with tf.device('/cpu:0'):
            all_files = [dump_file]
            print('tfrecords: ' + str(all_files))
            filename_queue = tf.train.string_input_producer(all_files, num_epochs=epochs ,shuffle=True)
            #example_list = [read_record(filename_queue) for _ in range(read_threads)]
            example_list = [read_fn(filename_queue) for _ in range(read_threads)]

            return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                         capacity=100 + batch_size * 16,
                                         min_after_dequeue=100,
                                         enqueue_many=False)


def read_record(filename_queue, reader):
    # reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)

    orig_image = features['image/encoded']

    oi1 = tf.image.decode_jpeg(orig_image)
    size = tf.minimum(img_h, img_w)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape)
    image = tf.image.resize_images(image, [128, 128])
    image = tf.reshape(image, (128, 128, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return img_h, img_w, crop_shape, image


def read_record_2(filename_queue, reader, scale):
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)

    orig_image = features['image/encoded']

    oi1 = tf.image.decode_jpeg(orig_image)
    size = tf.minimum(img_h, img_w)
    size = tf.cast(tf.round(tf.divide(tf.multiply(size, scale), 10)), tf.int32)
    size = tf.maximum(size, 128)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape)
    image = tf.image.resize_images(image, [128, 128])
    image = tf.reshape(image, (128, 128, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return img_h, img_w, crop_shape, image



if __name__ == '__main__':
    tf.app.run(argv=sys.argv)
