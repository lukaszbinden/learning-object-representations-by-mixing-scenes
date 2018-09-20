import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = '../datasets/coco/2017_val/tfrecords/train-00002-of-00003.tfrecords'  # address to save the hdf5 file

with tf.Session() as sess:
    feature={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
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

    # Reshape image data into the original shape
    # image = tf.reshape(image, [300, 300, 3])

    # Any preprocessing here ...
    #image = tf.image.decode_jpeg(image)
    #image = tf.reshape(image, [300, 300, 3])

    # Creates batches by randomly shuffling tensors
    #image = tf.train.shuffle_batch(image, batch_size=16, capacity=30, num_threads=1, min_after_dequeue=10)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    cnt = 0
    try:
        while not coord.should_stop():
            sess.run(image)
            cnt += 1
    except Exception as e:
        print('Done training -- epoch limit reached')
        print(e)
    finally:
        # Stop the threads
        coord.request_stop()
        coord.join(threads)

    sess.close()
    print('number of records: ' + str(cnt))
