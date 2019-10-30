import tensorflow as tf

# data_path = '/data/cvg/lukas/datasets/coco/2017_training/test_tfrecord/val-001-118287.tfrecords'
data_path = '../data/val-001-118287.tfrecords'
data_path = '/data/cvg/lukas/datasets/coco/2017_training/tfrecords_l2mix_flip_tile_10-L2nn_4285/181115/train-00011-of-00060.tfrecords'

with tf.Session() as sess:
    feature={'image/knn/t1': tf.VarLenFeature(tf.int64),
             'image/knn/t1s': tf.VarLenFeature(tf.int64),
             'image/knn/t1L2': tf.VarLenFeature(tf.float32),
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

    t2_10nn = features['image/knn/t1']
    t2_10nnd = tf.reshape(tf.sparse.to_dense(t2_10nn), (10,))
    t2_10nns = features['image/knn/t1s']
    t2_10nnds = tf.reshape(tf.sparse.to_dense(t2_10nns), (10,))
    t2_10nn_L2 = features['image/knn/t1L2']
    t2_10nnd_L2 = tf.reshape(tf.sparse.to_dense(t2_10nn_L2), (10,))

    ####################################################################################

    nn_id = tf.random_uniform([], 0, 9, dtype=tf.int32)

    t2_gather = tf.gather(t2_10nnd, nn_id)
    t2_one_nn = tf.as_string(t2_gather)
    t2_gathers = tf.gather(t2_10nnds, nn_id)
    t2_one_nns = tf.as_string(t2_gathers)

    postfix = tf.constant("_") + t2_one_nns + tf.constant("_t2.jpg")
    id_len = tf.strings.length(t2_one_nn)
    file_n = t2_one_nn + postfix

    z1 = tf.constant("0")
    z2 = tf.constant("00")
    z3 = tf.constant("000")
    z4 = tf.constant("0000")
    z5 = tf.constant("00000")
    z6 = tf.constant("000000")
    z7 = tf.constant("0000000")
    z8 = tf.constant("00000000")
    z9 = tf.constant("000000000")
    z10 = tf.constant("0000000000")
    z11 = tf.constant("00000000000")

    file_n = tf.where(tf.equal(id_len, 1), z11 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 2), z10 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 3), z9 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 4), z8 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 5), z7 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 6), z6 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 7), z5 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 8), z4 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 9), z3 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 10), z2 + file_n, file_n)
    file_n = tf.where(tf.equal(id_len, 11), z1 + file_n, file_n)

    # path = tf.constant("..\\data\\")
    # with tf.control_dependencies([tf.assert_equal(tf.strings.length(file_n), 21)]):
    #     file_n = path + file_n
    # # print('file_n: ', file_n)
    #
    # t2_file_nn = tf.read_file(file_n)
    # # print('t2_file_nn: ', t2_file_nn)
    # # print('i_ref_img: ', i_ref_img)
    #
    # t2_file_nn = tf.image.decode_jpeg(t2_file_nn)
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

            i_ref, id, t1_l2_img, l, t1_L2 = sess.run([i_ref_img, t2_one_nn, file_n, t2_10nnd, t2_10nnd_L2]) # t2_file,  t2_file_nn,
            print(type(i_ref))
            print(i_ref.shape)
            print(i_ref.size)
            print('--')
            print(id)
            print(t1_l2_img)
            print(l)
            # print('--')
            # print(type(t2_file))
            # print(t2_file.shape)
            # print(t2_file.size)
            print('--')
            print(type(t1_L2))
            print(t1_L2)
            print(type(t1_L2[0]))
            print(type(t1_L2[9]))

            print('-----------------<<')
            if times >= times_max:
                break


    finally:
        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()
