import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ast

data_path = '/data/cvg/lukas/datasets/coco/2017_training/test_tfrecord/val-001-118287.tfrecords'

with tf.Session() as sess:
    feature={'image/knn/t1': tf.FixedLenFeature([], tf.string),
             'image/encoded': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    #image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = features['image/encoded']
    t1_10nn_str = features['image/knn/t1']

    image = tf.image.decode_jpeg(image)




    #############################################################################
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    times_max = 2
    times = 0

    try:
        while not coord.should_stop():
            print('----------------->>')
            times = times + 1
            print('sess.run...')
            img, t1_10nn = sess.run([image, t1_10nn_str])
            # print(type(img))
            # print(img.shape)
            # print(img.size)
            print(type(t1_10nn))
            print(t1_10nn)
            t1_10nn = t1_10nn.decode("utf-8")
            print(type(t1_10nn))
            print(t1_10nn)
            t1_10nn = ast.literal_eval(t1_10nn)
            print(type(t1_10nn))
            print(t1_10nn)
            print('-----------------<<')
            if times >= times_max:
                break


    finally:
        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
