import glob
import sys
sys.path.append('..')
from squeezenet_model import squeezenet
from ops_alex import *

def main(_):
    with tf.Session() as sess:

        file = '../data/train-00010-of-00060.tfrecords'
        reader = tf.TFRecordReader()
        read_fn = lambda name : read_record_max(name, reader, 64)
        #h, w, crop_shape, train_images = get_pipeline(file, 4, 1, read_fn)

        filename, image, t1_10nn_ids, t1_10nn_subids, t1_10nn_L2, t2_10nn_ids, t2_10nn_subids, t2_10nn_L2, \
           t3_10nn_ids, t3_10nn_subids, t3_10nn_L2, t4_10nn_ids, t4_10nn_subids, t4_10nn_L2 = get_pipeline(file, 4, 1, read_fn)

        filename2, image2, _, _, _, _, _, _, _, _, _, _, _, _= get_pipeline(file, 4, 1, read_fn)

        with tf.variable_scope('generator') as scope_generator:

            concatenated1 = tf.concat(axis=3, values=[image, image])

            with tf.variable_scope('c_squeezenet'):
                tf.nn.sigmoid(squeezenet(concatenated1, num_classes=4))

            scope_generator.reuse_variables()

            concatenated = tf.concat(axis=3, values=[image, image2])
            with tf.variable_scope('c_squeezenet'):
                logits = squeezenet(concatenated, num_classes=4)
                assignments_predicted = tf.nn.sigmoid(logits)

        # a_actualE = tf.zeros((4, 4), dtype=tf.int32)
        a_actual = tf.constant([[0,1,0,0], [0,0,0,1], [1,0,0,1], [0,0,1,0]], dtype=tf.int32)

        with tf.variable_scope('classifier_loss'):
            cls_loss = binary_cross_entropy_with_logits(tf.cast(a_actual, tf.float32), assignments_predicted)

        t_vars = tf.trainable_variables()
        cls_vars = [var for var in t_vars if 'c_' in var.name] # classifier

        c_optim = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.5) \
            .minimize(cls_loss, var_list=cls_vars)  # params.beta1

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        try:
            iter = 0
            while not coord.should_stop():
                iter = iter + 1
                if iter == 1:
                    l, a_p = sess.run([logits, assignments_predicted])
                    print('logits 1st: %s' % str(l))
                    print('a_p 1st...: %s' % to_string(a_p))

                # tvars_vals = sess.run(cls_vars)
                # for var, val in zip(cls_vars, tvars_vals):
                #     print(var.name)  # Prints the name of the variable alongside its value.
                #     print(val)
                #     print("******************************")


                sess.run([c_optim])

                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                l, a_p = sess.run([logits, assignments_predicted])
                print('logits %d: %s' % (iter, str(l)))
                print('a_p...: %s' % to_string(a_p))

                # tvars_vals = sess.run(cls_vars)
                # for var, val in zip(cls_vars, tvars_vals):
                #     print(var.name)  # Prints the name of the variable alongside its value.
                #     print(val)
                #     print("******************************")


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


def to_string(ass_actual):
    st = ''
    for list in ass_actual:
        for e in list:
            st += str(e)
        st += '_'
    st = st[:-1]
    return st


def get_pipeline(dump_file, batch_size, epochs, read_fn, read_threads=4):
    with tf.variable_scope('dump_reader'):
        with tf.device('/cpu:0'):
            all_files = glob.glob(dump_file + '*')
            all_files = all_files if len(all_files) > 0 else [dump_file]
            print('tfrecords: ' + str(all_files))
            filename_queue = tf.train.string_input_producer(all_files, num_epochs=epochs ,shuffle=True, seed=4285)
            example_list = [read_fn(filename_queue) for _ in range(read_threads)]

            return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                         capacity=100 + batch_size * 16,
                                         min_after_dequeue=100,
                                         seed=4285,
                                         enqueue_many=False)

def get_pipeline_cherry(dump_file, batch_size, epochs, read_fn):
    with tf.variable_scope('dump_reader'):
        with tf.device('/cpu:0'):
            all_files = glob.glob(dump_file + '*')
            all_files = all_files if len(all_files) > 0 else [dump_file]
            print('tfrecords: ' + str(all_files))
            filename_queue = tf.train.string_input_producer(all_files, num_epochs=epochs ,shuffle=True, seed=4285)
            example = read_fn(filename_queue)

            return tf.train.batch(example, batch_size=batch_size,
                                         capacity=100 + batch_size * 16,
                                         enqueue_many=False)


def read_record_scale(filename_queue, reader, image_size, scale, crop=True):
    _, serialized_example = reader.read(filename_queue)

    print("read_record_scale...")

    features = tf.parse_single_example(
      serialized_example,
      features={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/knn/t1': tf.VarLenFeature(tf.int64),
                'image/knn/t1s': tf.VarLenFeature(tf.int64),
                'image/knn/t1L2': tf.VarLenFeature(tf.float32),
                'image/knn/t2': tf.VarLenFeature(tf.int64),
                'image/knn/t2s': tf.VarLenFeature(tf.int64),
                'image/knn/t2L2': tf.VarLenFeature(tf.float32),
                'image/knn/t3': tf.VarLenFeature(tf.int64),
                'image/knn/t3s': tf.VarLenFeature(tf.int64),
                'image/knn/t3L2': tf.VarLenFeature(tf.float32),
                'image/knn/t4': tf.VarLenFeature(tf.int64),
                'image/knn/t4s': tf.VarLenFeature(tf.int64),
                'image/knn/t4L2': tf.VarLenFeature(tf.float32),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)
    filename = features['image/filename']

    t1_10nn_ids = features['image/knn/t1']
    t1_10nn_subids = features['image/knn/t1s']
    t1_10nn_L2 = features['image/knn/t1L2']
    t2_10nn_ids = features['image/knn/t2']
    t2_10nn_subids = features['image/knn/t2s']
    t2_10nn_L2 = features['image/knn/t2L2']
    t3_10nn_ids = features['image/knn/t3']
    t3_10nn_subids = features['image/knn/t3s']
    t3_10nn_L2 = features['image/knn/t3L2']
    t4_10nn_ids = features['image/knn/t4']
    t4_10nn_subids = features['image/knn/t4s']
    t4_10nn_L2 = features['image/knn/t4L2']

    orig_image = features['image/encoded']

    oi1 = tf.image.decode_jpeg(orig_image)
    if crop:
        size = tf.minimum(img_h, img_w)
        if scale:
            size = tf.cast(tf.round(tf.divide(tf.multiply(size, scale), 10)), tf.int32)
        size = tf.maximum(size, image_size)
        crop_shape = tf.parallel_stack([size, size, 3])
        image = tf.random_crop(oi1, crop_shape, seed=4285)
    else:
        image = oi1
    image = tf.image.resize_images(image, [image_size, image_size])
    image = tf.reshape(image, (image_size, image_size, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return filename, image, t1_10nn_ids, t1_10nn_subids, t1_10nn_L2, t2_10nn_ids, t2_10nn_subids, t2_10nn_L2, \
           t3_10nn_ids, t3_10nn_subids, t3_10nn_L2, t4_10nn_ids, t4_10nn_subids, t4_10nn_L2


def read_record_max(filename_queue, reader, image_size, crop=True):
    return read_record_scale(filename_queue, reader, image_size, None, crop)

if __name__ == '__main__':
    tf.app.run(argv=sys.argv)
