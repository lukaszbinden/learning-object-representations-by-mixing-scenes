import tensorflow as tf
import glob


data_path = '/data/cvg/imagenet/imagenet_tfrecords/train-00383-of-01024'
name_image_feature = 'image/encoded'
name_image_feature = 'encoded'

with tf.Session() as sess:

    for example in tf.python_io.tf_record_iterator(data_path):
        result = tf.train.Example.FromString(example)
        break

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            r = sess.run(result)
            print(r)
    except Exception as e:
        print('Done training -- epoch limit reached')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(e)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    finally:
        # Stop the threads
        coord.request_stop()
        coord.join(threads)

    sess.close()

    print("done.")
