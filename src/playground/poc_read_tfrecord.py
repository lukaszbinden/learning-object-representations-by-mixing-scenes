import tensorflow as tf

# data_path = '/data/cvg/lukas/datasets/coco/2017_training/test_tfrecord/val-001-118287.tfrecords'
data_path = '../data/val-001-118287.tfrecords'

with tf.Session() as sess:
    feature={'image/knn/t1': tf.FixedLenFeature([], tf.string),
             'image/knn/t2': tf.VarLenFeature(tf.int64),
             'image/encoded': tf.FixedLenFeature([], tf.string)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    #image = tf.decode_raw(features['image/encoded'], tf.uint8)
    i_ref_img = features['image/encoded']

    t1_10nn_str = features['image/knn/t1']
    print('t1_10nn_str: ', type(t1_10nn_str))
    # print('t1_10nn_str[0]: ', type(t1_10nn_str[0]))

    t2_10nn = features['image/knn/t2']
    print('t2_10nn: ', type(t2_10nn))
    t2_10nnd = tf.reshape(tf.sparse.to_dense(t2_10nn), (10,))
    print('t2_10nnd.shape: ', t2_10nnd.shape)

    nn_id = tf.random_uniform([], 0, 9, dtype=tf.int32)

    t2_gather = tf.gather(t2_10nnd, nn_id)
    t2_one_nn = tf.as_string(t2_gather)
    print('t2_one_nn: ', t2_one_nn)
    prefix = tf.constant("..\\data\\000000")
    postfix = tf.constant("_1_t2.jpg")
    # file_n = tf.strings.format("000000{}_1_t2.jpg", t2_one_nn)
    file_n = prefix + t2_one_nn + postfix
    print('file_n: ', file_n)

    t2_file_nn = tf.read_file(file_n)
    print('t2_file_nn: ', t2_file_nn)
    print('i_ref_img: ', i_ref_img)

    # d = tf.data.Dataset.from_tensors(tf.constant(1))
    # def load_file(_):
    #     return tf.read_file(file_n)
    # d = d.map(load_file)
    # iterator = d.make_one_shot_iterator()
    # t2_file_nn = iterator.get_next()

    t2_file_nn = tf.image.decode_jpeg(t2_file_nn)
    i_ref_img = tf.image.decode_jpeg(i_ref_img)

    #############################################################################
    t1_10nn1 = [('000000000927_1.jpg', 0.03125), ('000000568135_2.jpg', 19095.953), ('000000187857_1.jpg', 23359.39),
               ('000000521998_2.jpg', 23557.688), ('000000140816_1.jpg', 24226.852), ('000000015109_1.jpg', 25191.469),
               ('000000525567_1.jpg', 25484.93), ('000000377422_1.jpg', 25654.125), ('000000269815_2.jpg', 26794.836),
               ('000000345617_2.jpg', 26872.812)]
    #############################################################################
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    times_max = 1
    times = 0

    try:
        while not coord.should_stop():
            print('----------------->>')
            times = times + 1
            print('sess.run...')

            i_ref, id, t1_l2_img, l, t2_file = sess.run([i_ref_img, t2_one_nn, file_n, t2_10nnd, t2_file_nn])
            print(type(i_ref))
            print(i_ref.shape)
            print(i_ref.size)
            print('--')
            print(id)
            print(t1_l2_img)
            print(l)
            print('--')
            print(type(t2_file))
            print(t2_file.shape)
            print(t2_file.size)

            print('-----------------<<')
            if times >= times_max:
                break


    finally:
        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
