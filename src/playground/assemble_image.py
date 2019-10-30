import tensorflow as tf
import glob
from scipy.misc import imsave

# data_path = '../datasets/coco/2017_training/tfrecords/'  # address to save the hdf5 file
data_path = '../data/2017_train_small_anys.tfrecords'  # address to save the hdf5 file

with tf.Session() as sess:
    feature={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string)}

    all_files = glob.glob(data_path + '*')
    all_files = all_files if len(all_files) > 0 else [data_path]
    print('counting in %s files...' % (len(all_files)))
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(all_files, num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    _, serialized_example2 = reader.read(filename_queue)
    _, serialized_example3 = reader.read(filename_queue)
    _, serialized_example4 = reader.read(filename_queue)
    exs = [serialized_example, serialized_example2, serialized_example3, serialized_example4]

    # Decode the record read by the reader
    features = tf.parse_example(exs, features=feature)

    #image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = features['image/encoded']
    name = features['image/filename']
    img_h = features['image/height']
    img_w = features['image/width']
    img_h = tf.cast(img_h, tf.int32)
    img_w = tf.cast(img_w, tf.int32)

    # TODO try same but which batch_size = 4 ie tensor of dim 4

    img = tf.image.decode_jpeg(image[0])
    img_shape = tf.parallel_stack([img_h[0], img_w[0], 3])
    image1 = tf.reshape(img, img_shape)
    img = tf.image.decode_jpeg(image[1])
    img_shape = tf.parallel_stack([img_h[1], img_w[1], 3])
    image2 = tf.reshape(img, img_shape)
    img = tf.image.decode_jpeg(image[2])
    img_shape = tf.parallel_stack([img_h[2], img_w[2], 3])
    image3 = tf.reshape(img, img_shape)
    img = tf.image.decode_jpeg(image[3])
    img_shape = tf.parallel_stack([img_h[3], img_w[3], 3])
    image4 = tf.reshape(img, img_shape)

    t1 = tf.image.crop_to_bounding_box(image1, 0, 0, 64, 64)
    t1 = tf.reshape(t1, (64,64,3))
    t2 = tf.image.crop_to_bounding_box(image2, 0, 0, 64, 64)
    t2 = tf.reshape(t2, (64,64,3))
    t3 = tf.image.crop_to_bounding_box(image3, 0, 0, 64, 64)
    t3 = tf.reshape(t3, (64,64,3))
    t4 = tf.image.crop_to_bounding_box(image4, 0, 0, 64, 64)
    t4 = tf.reshape(t4, (64,64,3))

    t1 = tf.expand_dims(t1, 0)
    t2 = tf.expand_dims(t2, 0)
    t3 = tf.expand_dims(t3, 0)
    t4 = tf.expand_dims(t4, 0)

    f = tf.concat([t1, t3], axis=1)
    f2 = tf.concat([t2, t4], axis=1)
    f4 = tf.concat([f, f2], axis=2)
    print(f4.shape)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    cnt = 0
    try:
        while not coord.should_stop():
            r1, r2, r3, r4, rf,rf2,rf4 = sess.run([t1, t2,t3,t4,f, f2,f4])
            imsave('r1.jpeg', r1[0])
            imsave('r2.jpeg', r2[0])
            imsave('r3.jpeg', r3[0])
            imsave('r4.jpeg', r4[0])
            imsave('rf.jpeg', rf[0])
            imsave('rf2.jpeg', rf2[0])
            imsave('rf4.jpeg', rf4[0])
            break


    except Exception as e:
        if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done iterating -- all data processed.')
        else:
            print('Exception here, ending loop..')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(e)
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    finally:
        # Stop the threads
        coord.request_stop()
        coord.join(threads)

    sess.close()
    print('number of records: %s in %s files.' % (str(cnt), len(all_files)))
