import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ast

data_path = '/data/cvg/lukas/datasets/coco/2017_training/test_tfrecord/val-001-118287.tfrecords'
data_path = '/data/cvg/imagenet/imagenet_tfrecords/train-00383-of-01024'
data_path = '/data/cvg/imagenet/imagenet_tfrecords/validation-00122-of-00128'
data_path = '../data/imagenet_00122-of-00128.tfrecords'
name_image_feature = 'image/encoded'
name_image_feature = 'encoded'

with tf.Session() as sess:
    feature={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/channels': tf.FixedLenFeature([], tf.int64),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)


    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)
    img_ch = features['image/channels']
    img_ch = tf.cast(img_ch, tf.int32)
    class_id = features['image/class/label']
    orig_image = features['image/encoded']
    fname = features['image/filename']

    oi1 = tf.image.decode_jpeg(orig_image, channels=3)

    # oi1 = tf.cond(tf.equal(img_ch, 1),
    #          true_fn=lambda: tf.image.grayscale_to_rgb(oi1), false_fn=lambda: oi1)



    #############################################################################
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    times_max = 1000000
    times = 0

    try:
        while not coord.should_stop():
            print('----------------->>')
            times = times + 1
            print('sess.run...')
            img_h1, img_w1, img_ch1, class_id1, oi11, fname1 = sess.run([img_h, img_w, img_ch, class_id, oi1, fname])

            print(img_h1)
            print(img_w1)
            print(img_ch1)
            print(class_id1)
            print(oi11.shape)

            if img_ch1 == 1 or img_ch1 != oi11.shape[2]:
                print("found gray!!!!!!")
                print(img_ch1)
                print("fname:", fname1.decode("utf-8"))
                print(oi11.shape)
                break

            print('-----------------<<')
            if times >= times_max:
                break


    finally:
        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()

