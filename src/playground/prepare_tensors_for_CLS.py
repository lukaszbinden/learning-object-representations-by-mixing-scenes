import tensorflow as tf
import glob
import sys
import os
from scipy.misc import imsave
import traceback
from datetime import datetime

def main(_):
    with tf.Session() as sess:

        #tf.set_random_seed(4285)

        epochs = 1
        batch_size = 4  # must divide dataset size (some strange error occurs if not)
        image_size = 128

        tfrecords_file_in = '../data/train-00011-of-00060.tfrecords'  # '/data/cvg/lukas/datasets/coco/2017_training/tfrecords_l2mix_flip_tile_10-L2nn_4285/181115/'  #
        filedir_out_base = '../logs/test/test_tfrecords_with_tile_10L2nn'
        tile_filedir_in = '/data/cvg/lukas/datasets/coco/2017_training/clustering_224x224_4285/'
        tile_filedir_out = '~/results/knn_results/'

        reader = tf.TFRecordReader()
        read_fn = lambda name : read_record(name, reader, image_size)
        # filename, train_images, t1_10nn_ids, t2_10nn_ids, t3_10nn_ids, t4_10nn_ids, t1_10nn_subids, t2_10nn_subids, t3_10nn_subids, t4_10nn_subids = get_pipeline(tfrecords_file_in, batch_size, epochs, read_fn)
        filenames, train_images, t1_10nn_ids, t1_10nn_subids, t1_10nn_L2, t2_10nn_ids, t2_10nn_subids, t2_10nn_L2, t3_10nn_ids, t3_10nn_subids, t3_10nn_L2, t4_10nn_ids, t4_10nn_subids, t4_10nn_L2 = \
            get_pipeline(tfrecords_file_in, batch_size, epochs, read_fn)

        images_I_ref = train_images

        print('t1_10nn_ids ', t1_10nn_ids)
        t1_10nn_ids = tf.reshape(tf.sparse.to_dense(t1_10nn_ids), (batch_size, -1))
        print('t1_10nn_ids ', t1_10nn_ids)
        t1_10nn_L2 = tf.reshape(tf.sparse.to_dense(t1_10nn_L2), (batch_size, -1))
        print('t1_10nn_L2 ', t1_10nn_L2)
        t1_10nn_subids = tf.reshape(tf.sparse.to_dense(t1_10nn_subids), (batch_size, -1))
        t2_10nn_ids = tf.reshape(tf.sparse.to_dense(t2_10nn_ids), (batch_size, -1))
        t2_10nn_L2 = tf.reshape(tf.sparse.to_dense(t2_10nn_L2), (batch_size, -1))
        t2_10nn_subids = tf.reshape(tf.sparse.to_dense(t2_10nn_subids), (batch_size, -1))
        t3_10nn_ids = tf.reshape(tf.sparse.to_dense(t3_10nn_ids), (batch_size, -1))
        t3_10nn_subids = tf.reshape(tf.sparse.to_dense(t3_10nn_subids), (batch_size, -1))
        t3_10nn_L2 = tf.reshape(tf.sparse.to_dense(t3_10nn_L2), (batch_size, -1))
        t4_10nn_ids = tf.reshape(tf.sparse.to_dense(t4_10nn_ids), (batch_size, -1))
        t4_10nn_subids = tf.reshape(tf.sparse.to_dense(t4_10nn_subids), (batch_size, -1))
        t4_10nn_L2 = tf.reshape(tf.sparse.to_dense(t4_10nn_L2), (batch_size, -1))

        nn_id = tf.random_uniform([batch_size], 0, 9, dtype=tf.int32)

        tile_size = image_size / 2
        assert tile_size.is_integer()
        tile_size = int(tile_size)

        underscore = tf.constant("_")
        path = tf.constant("/data/cvg/lukas/datasets/coco/2017_training/clustering_224x224_4285/")
        # t1 ############################################################################################
        path_prefix_t1 = path + tf.constant("t1/")
        filetype = tf.constant("_t1.jpg")
        for id in range(batch_size):
            t1_10nn_ids_b = t1_10nn_ids[id]
            index = nn_id[id]
            t1_10nn_id = tf.gather(t1_10nn_ids_b, index)
            t1_10nn_id_str = tf.as_string(t1_10nn_id)
            t1_10nn_subids_b = t1_10nn_subids[id]
            t1_10nn_subid = tf.gather(t1_10nn_subids_b, index)
            t1_10nn_subid_str = tf.as_string(t1_10nn_subid)
            postfix = underscore + t1_10nn_subid_str + filetype
            fname = get_filename(t1_10nn_id_str, postfix)
            t1_10nn_fnames = fname if id == 0 else tf.concat(axis=0, values=[t1_10nn_fnames, fname])

        with tf.control_dependencies([tf.assert_equal(batch_size, t1_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t1_10nn_fnames), 21)]):
            print(t1_10nn_fnames.shape)
            t1_10nn_fnames = tf.strings.join([path_prefix_t1, t1_10nn_fnames])
            print('<<<<<<<<<<<<<<<<<<<')
            print(t1_10nn_fnames.shape)
            print('<<<<<<<<<<<<<<<<<<<')
            print('t1_10nn_fnames.shape: %s' % str(t1_10nn_fnames.shape))

            for id in range(batch_size):
                file = tf.read_file(t1_10nn_fnames[id])
                print(file)
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, tile_size, batch_size)
                file = tf.expand_dims(file, 0)
                t1_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t1_10nn_images, file])

        print('train_images.shape..:', train_images.shape)
        print('t1_10nn_images.shape:', t1_10nn_images.shape)

        # t2 ############################################################################################
        path_prefix_t2 = path + tf.constant("t2/")
        filetype = tf.constant("_t2.jpg")
        for id in range(batch_size):
            t2_10nn_ids_b = t2_10nn_ids[id]
            index = nn_id[id]
            t2_10nn_id = tf.gather(t2_10nn_ids_b, index)
            t2_10nn_id_str = tf.as_string(t2_10nn_id)
            t2_10nn_subids_b = t2_10nn_subids[id]
            t2_10nn_subid = tf.gather(t2_10nn_subids_b, index)
            t2_10nn_subid_str = tf.as_string(t2_10nn_subid)
            postfix = underscore + t2_10nn_subid_str + filetype
            fname = get_filename(t2_10nn_id_str, postfix)
            t2_10nn_fnames = fname if id == 0 else tf.concat(axis=0, values=[t2_10nn_fnames, fname])

        with tf.control_dependencies([tf.assert_equal(batch_size, t2_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t2_10nn_fnames), 21)]):
            print(t2_10nn_fnames.shape)
            t2_10nn_fnames = tf.strings.join([path_prefix_t2, t2_10nn_fnames])
            print('<<<<<<<<<<<<<<<<<<<')
            print(t2_10nn_fnames.shape)
            print('<<<<<<<<<<<<<<<<<<<')
            print('t2_10nn_fnames.shape: %s' % str(t2_10nn_fnames.shape))

            for id in range(batch_size):
                file = tf.read_file(t2_10nn_fnames[id])
                print(file)
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, tile_size, batch_size)
                file = tf.expand_dims(file, 0)
                t2_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t2_10nn_images, file])

        print('train_images.shape..:', train_images.shape)
        print('t2_10nn_images.shape:', t2_10nn_images.shape)

        # t3 ############################################################################################
        path_prefix_t3 = path + tf.constant("t3/")
        filetype = tf.constant("_t3.jpg")
        for id in range(batch_size):
            t3_10nn_ids_b = t3_10nn_ids[id]
            index = nn_id[id]
            t3_10nn_id = tf.gather(t3_10nn_ids_b, index)
            t3_10nn_id_str = tf.as_string(t3_10nn_id)
            t3_10nn_subids_b = t3_10nn_subids[id]
            t3_10nn_subid = tf.gather(t3_10nn_subids_b, index)
            t3_10nn_subid_str = tf.as_string(t3_10nn_subid)
            postfix = underscore + t3_10nn_subid_str + filetype
            fname = get_filename(t3_10nn_id_str, postfix)
            t3_10nn_fnames = fname if id == 0 else tf.concat(axis=0, values=[t3_10nn_fnames, fname])

        with tf.control_dependencies([tf.assert_equal(batch_size, t3_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t3_10nn_fnames), 21)]):
            print(t3_10nn_fnames.shape)
            t3_10nn_fnames = tf.strings.join([path_prefix_t3, t3_10nn_fnames])
            print('<<<<<<<<<<<<<<<<<<<')
            print(t3_10nn_fnames.shape)
            print('<<<<<<<<<<<<<<<<<<<')
            print('t3_10nn_fnames.shape: %s' % str(t3_10nn_fnames.shape))

            for id in range(batch_size):
                file = tf.read_file(t3_10nn_fnames[id])
                print(file)
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, tile_size, batch_size)
                file = tf.expand_dims(file, 0)
                t3_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t3_10nn_images, file])

        print('train_images.shape..:', train_images.shape)
        print('t3_10nn_images.shape:', t3_10nn_images.shape)

        # t4 ############################################################################################
        path_prefix_t4 = path + tf.constant("t4/")
        filetype = tf.constant("_t4.jpg")
        for id in range(batch_size):
            t4_10nn_ids_b = t4_10nn_ids[id]
            index = nn_id[id]
            t4_10nn_id = tf.gather(t4_10nn_ids_b, index)
            t4_10nn_id_str = tf.as_string(t4_10nn_id)
            t4_10nn_subids_b = t4_10nn_subids[id]
            t4_10nn_subid = tf.gather(t4_10nn_subids_b, index)
            t4_10nn_subid_str = tf.as_string(t4_10nn_subid)
            postfix = underscore + t4_10nn_subid_str + filetype
            fname = get_filename(t4_10nn_id_str, postfix)
            t4_10nn_fnames = fname if id == 0 else tf.concat(axis=0, values=[t4_10nn_fnames, fname])

        with tf.control_dependencies([tf.assert_equal(batch_size, t4_10nn_fnames.shape[0]), tf.assert_equal(tf.strings.length(t4_10nn_fnames), 21)]):
            print(t4_10nn_fnames.shape)
            t4_10nn_fnames = tf.strings.join([path_prefix_t4, t4_10nn_fnames])
            print('<<<<<<<<<<<<<<<<<<<')
            print(t4_10nn_fnames.shape)
            print('<<<<<<<<<<<<<<<<<<<')
            print('t4_10nn_fnames.shape: %s' % str(t4_10nn_fnames.shape))

            for id in range(batch_size):
                file = tf.read_file(t4_10nn_fnames[id])
                print(file)
                file = tf.image.decode_jpeg(file)
                file = resize_img(file, tile_size, batch_size)
                file = tf.expand_dims(file, 0)
                t4_10nn_images = file if id == 0 else tf.concat(axis=0, values=[t4_10nn_images, file])

        print('train_images.shape..:', train_images.shape)
        print('t4_10nn_images.shape:', t4_10nn_images.shape)

        # ###########################################################################################################
        # ###########################################################################################################

        I_ref_t1 = tf.image.crop_to_bounding_box(images_I_ref, 0, 0, tile_size, tile_size)
        I_ref_t2 = tf.image.crop_to_bounding_box(images_I_ref, 0, tile_size, tile_size, tile_size)
        I_ref_t3 = tf.image.crop_to_bounding_box(images_I_ref, tile_size, 0, tile_size, tile_size)
        I_ref_t4 = tf.image.crop_to_bounding_box(images_I_ref, tile_size, tile_size, tile_size, tile_size)

        # self.images_I_ref
        # self.I_ref_t1
        # self.I_ref_t2
        # self.I_ref_t3
        # self.I_ref_t4
        #
        # self.images_t1
        # self.images_t2
        # self.images_t3
        # self.images_t4
        #
        # self.images_
        # self.images_t2
        # self.images_t3
        # self.images_t4
        #
        # t1_10nn_L2
        #
        # # replace tile w/ max L2 wrt I_ref w/ respective tile of I_ref
        # # TODO: assign tiles to these
        J_1_tile = None
        J_2_tile = None
        J_3_tile = None
        J_4_tile = None
        assignments_actual = tf.zeros((batch_size, 4, 4))


        # a = tf.get_variable("assign1", dtype=tf.int32, initializer=tf.constant([1, 1, 1, 1]))

        # TODO: ultimately, we want this:
        # f_I1_I2_mix
        for id in range(batch_size):
            t1_10nn_L2_b = t1_10nn_L2[id]
            index = nn_id[id]
            t1_10nn_L2_b = tf.gather(t1_10nn_L2_b, index)
            t2_10nn_L2_b = t2_10nn_L2[id]
            t2_10nn_L2_b = tf.gather(t2_10nn_L2_b, index)
            t3_10nn_L2_b = t3_10nn_L2[id]
            t3_10nn_L2_b = tf.gather(t3_10nn_L2_b, index)
            t4_10nn_L2_b = t4_10nn_L2[id]
            t4_10nn_L2_b = tf.gather(t4_10nn_L2_b, index)
            all_L2 = tf.stack(axis=0, values=[t1_10nn_L2_b, t2_10nn_L2_b, t3_10nn_L2_b, t4_10nn_L2_b])
            all_L2 = tf.reshape(all_L2, [-1])
            argmax_L2 = tf.argmax(all_L2, axis=0)

            # replace the tile that has max L2 with tile from I_ref
            tile_1 = tf.expand_dims(tf.where(tf.equal(argmax_L2, 0), I_ref_t1[id], t1_10nn_images[id]), 0)
            assignment_1 = tf.where(tf.equal(argmax_L2, 0), 0, 1)
            J_1_tile = tile_1 if id == 0 else tf.concat(axis=0, values=[J_1_tile, tile_1])
            tile_2 = tf.expand_dims(tf.where(tf.equal(argmax_L2, 1), I_ref_t2[id], t2_10nn_images[id]), 0)
            assignment_2 = tf.where(tf.equal(argmax_L2, 1), 0, 1)
            J_2_tile = tile_2 if id == 0 else tf.concat(axis=0, values=[J_2_tile, tile_2])
            tile_3 = tf.expand_dims(tf.where(tf.equal(argmax_L2, 2), I_ref_t3[id], t3_10nn_images[id]), 0)
            assignment_3 = tf.where(tf.equal(argmax_L2, 2), 0, 1)
            J_3_tile = tile_3 if id == 0 else tf.concat(axis=0, values=[J_3_tile, tile_3])
            tile_4 = tf.expand_dims(tf.where(tf.equal(argmax_L2, 3), I_ref_t4[id], t4_10nn_images[id]), 0)
            assignment_4 = tf.where(tf.equal(argmax_L2, 3), 0, 1)
            J_4_tile = tile_4 if id == 0 else tf.concat(axis=0, values=[J_4_tile, tile_4])

            # TODO: also replace tiles with I_ref where L2 > tau (threshold)
            # TODO: enusre tile with least L2 remains selected

            assignments = tf.stack(axis=0, values=[assignment_1, assignment_2, assignment_3, assignment_4])
            assignments = tf.reshape(assignments, [-1])
            assignments = tf.expand_dims(assignments, 0)
            assignments_actual = assignments if id == 0 else tf.concat(axis=0, values=[assignments_actual, assignments])

        assert J_1_tile.shape[0] == batch_size
        assert J_1_tile.shape[1] == tile_size
        assert J_1_tile.shape[2] == tile_size
        assert J_1_tile.shape[3] == 3
        assert J_1_tile.shape == J_2_tile.shape
        assert J_2_tile.shape == J_3_tile.shape
        assert J_2_tile.shape == J_4_tile.shape








        # [('000000000927_1.jpg', 0.03125), ('000000568135_2.jpg', 19095.953), ('000000187857_1.jpg', 23359.39),
        #  ('000000521998_2.jpg', 23557.688), ('000000140816_1.jpg', 24226.852), ('000000015109_1.jpg', 25191.469),
        #  ('000000525567_1.jpg', 25484.93), ('000000377422_1.jpg', 25654.125), ('000000269815_2.jpg', 26794.836),
        #  ('000000345617_2.jpg', 26872.812)]

        ########################################################################################################

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        max_batches = 5
        cnt_batches = 0

        max_iterations = batch_size
        timef = datetime.now().strftime('%Y%m%d_%H%M%S')
        filedir_out_base = os.path.join(filedir_out_base, timef)

        try:
            while not coord.should_stop():

                # r, s = sess.run([t1_10nn_ids, t1_10nn_subids])
                # print(r)
                # print(s)

                print('assignments_actual.shape: ', assignments_actual.shape)

                aa = sess.run([assignments_actual])

                print(aa)

                cnt_iterations = 0
                for i in range(batch_size):
                    print('ITERATION [%d] >>>>>>' % i)

                    print('ITERATION [%d] <<<<<<' % i)
                    cnt_iterations = cnt_iterations + 1
                    if cnt_iterations >= max_iterations:
                        break

                cnt_batches = cnt_batches + 1
                if cnt_batches >= max_batches:
                    break


        except Exception as e:
            if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Exception here, ending training..')
                tb = traceback.format_exc()
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(e)
                print(tb)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def get_filename(t2_one_nn, postfix):
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

    return tf.expand_dims(file_n, 0)


def get_pipeline(dump_file, batch_size, epochs, read_fn, read_threads=4):
    with tf.variable_scope('dump_reader'):
        with tf.device('/cpu:0'):
            all_files = glob.glob(dump_file + '*')
            all_files = all_files if len(all_files) > 0 else [dump_file]
            print('tfrecords: ' + str(all_files))
            filename_queue = tf.train.string_input_producer(all_files, num_epochs=epochs ,shuffle=True)
            #example_list = [read_record(filename_queue) for _ in range(read_threads)]
            example_list = [read_fn(filename_queue) for _ in range(read_threads)]

            return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                         capacity=100 + batch_size * 16,
                                         min_after_dequeue=100,
                                         enqueue_many=False) #,
                                         #allow_smaller_final_batch=True)


def read_record(filename_queue, reader, img_size):
    # reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

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

    image = preprocess_image(orig_image, img_size, img_w, img_h)

    return filename, image, t1_10nn_ids, t1_10nn_subids, t1_10nn_L2, t2_10nn_ids, t2_10nn_subids, t2_10nn_L2, \
           t3_10nn_ids, t3_10nn_subids, t3_10nn_L2, t4_10nn_ids, t4_10nn_subids, t4_10nn_L2


def preprocess_image(orig_image, img_size, img_w, img_h):
    oi1 = tf.image.decode_jpeg(orig_image)
    size = tf.minimum(img_h, img_w)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape)
    image = tf.image.resize_images(image, [img_size, img_size])
    image = tf.reshape(image, (img_size, img_size, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1
    return image


def resize_img(image, img_size, batch_size):
    image = tf.image.resize_images(image, [img_size, img_size])
    if len(image.shape) == 4: # check for batch case
        image = tf.reshape(image, (batch_size, img_size, img_size, 3))
        with tf.control_dependencies([tf.assert_equal(batch_size, image.shape[0])]):
            return image
    else:
        return tf.reshape(image, (img_size, img_size, 3))


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)


