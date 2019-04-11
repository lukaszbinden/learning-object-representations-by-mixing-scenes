import tensorflow as tf
import glob

data_path = '../datasets/coco/2017_training/tfrecords/'  # address to save the hdf5 file
# data_path = '/data/cvg/lukas/datasets/coco/2017_test/version/v2/final/'
data_path = '/data/cvg/lukas/datasets/coco/2017_training/version/v6/tmp/'
data_path = '/data/cvg/lukas/datasets/coco/2017_training/version/v6/final/'
data_path = '/data/cvg/lukas/datasets/coco/2017_test/version/v3/final/'
data_path = '/home/lz01a008/git/yad2k/YAD2K/voc_conversion_scripts/VOCdevkit/tfrecords/train/'
name_image_feature = 'image/encoded'
name_image_feature = 'encoded'

with tf.Session() as sess:
    feature={name_image_feature: tf.FixedLenFeature([], tf.string)}

    all_files = glob.glob(data_path + '*')
    all_files = all_files if len(all_files) > 0 else [data_path]
    print('counting in %s files...' % (len(all_files)))
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(all_files, num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    #image = tf.decode_raw(features['image/encoded'], tf.uint8)
    print(features)
    image = features[name_image_feature]

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
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(e)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    finally:
        # Stop the threads
        coord.request_stop()
        coord.join(threads)

    sess.close()
    print('number of records: %s in %s files.' % (str(cnt), len(all_files)))
