import tensorflow as tf
import glob
import sys
import os
from scipy.misc import imsave
import traceback

def main(_):
    with tf.Session() as sess:

        tf.set_random_seed(4285)

        epochs = 1
        batch_size = 4 # must divide dataset size (some strange error occurs if not)
        image_size = 128

        tfrecords_file_in = 'data/val-001-118287.tfrecords'  # ''/data/cvg/lukas/datasets/coco/2017_training/tfrecords_l2mix_flip_tile_10-L2nn_4285/'
        filedir_out = '../logs/test/test_tfrecords_with_tile_10L2nn'
        tile_filedir_in = '/data/cvg/lukas/datasets/coco/2017_training/clustering_224x224_4285/'
        tile_filedir_out = '~/results/knn_results/'

        reader = tf.TFRecordReader()
        read_fn = lambda name : read_record(name, reader, image_size)
        filename, train_images, t1_10nn_str, t2_10nn_str, t3_10nn_str, t4_10nn_str, \
                t1_10nn_strs, t2_10nn_strs, t3_10nn_strs, t4_10nn_strs = \
                get_pipeline(tfrecords_file_in, batch_size, epochs, read_fn)

        t1_10nn_str = tf.reshape(tf.sparse.to_dense(t1_10nn_str), (batch_size, 10))
        t2_10nn_str = tf.reshape(tf.sparse.to_dense(t2_10nn_str), (batch_size, 10))
        t3_10nn_str = tf.reshape(tf.sparse.to_dense(t3_10nn_str), (batch_size, 10))
        t4_10nn_str = tf.reshape(tf.sparse.to_dense(t4_10nn_str), (batch_size, 10))

        t1_10nn_strs = tf.reshape(tf.sparse.to_dense(t1_10nn_strs), (batch_size, 10))
        t2_10nn_strs = tf.reshape(tf.sparse.to_dense(t2_10nn_strs), (batch_size, 10))
        t3_10nn_strs = tf.reshape(tf.sparse.to_dense(t3_10nn_strs), (batch_size, 10))
        t4_10nn_strs = tf.reshape(tf.sparse.to_dense(t4_10nn_strs), (batch_size, 10))

        underscore = tf.constant("_")
        # t1
        filetype = tf.constant("_t1.jpg")
        for nn_id in range(batch_size):
            t2_gather = tf.gather(t1_10nn_str, nn_id)
            t2_one_nn = tf.as_string(t2_gather)
            t2_gathers = tf.gather(t1_10nn_strs, nn_id)
            t2_one_nns = tf.as_string(t2_gathers)
            postfix = underscore + t2_one_nns + filetype
            fname = get_filename(t2_one_nn, postfix)
            t1_10nn_fnames = fname if nn_id == 0 else tf.concat(axis=0, values=[t1_10nn_fnames, fname])
        # t2
        filetype = tf.constant("_t2.jpg")
        for nn_id in range(batch_size):
            t2_gather = tf.gather(t2_10nn_str, nn_id)
            t2_one_nn = tf.as_string(t2_gather)
            t2_gathers = tf.gather(t2_10nn_strs, nn_id)
            t2_one_nns = tf.as_string(t2_gathers)
            postfix = underscore + t2_one_nns + filetype
            fname = get_filename(t2_one_nn, postfix)
            t2_10nn_fnames = fname if nn_id == 0 else tf.concat(axis=0, values=[t2_10nn_fnames, fname])
        # t3
        filetype = tf.constant("_t3.jpg")
        for nn_id in range(batch_size):
            t2_gather = tf.gather(t3_10nn_str, nn_id)
            t2_one_nn = tf.as_string(t2_gather)
            t2_gathers = tf.gather(t3_10nn_strs, nn_id)
            t2_one_nns = tf.as_string(t2_gathers)
            postfix = underscore + t2_one_nns + filetype
            fname = get_filename(t2_one_nn, postfix)
            t3_10nn_fnames = fname if nn_id == 0 else tf.concat(axis=0, values=[t3_10nn_fnames, fname])
        # t4
        filetype = tf.constant("_t4.jpg")
        for nn_id in range(batch_size):
            t2_gather = tf.gather(t4_10nn_str, nn_id)
            t2_one_nn = tf.as_string(t2_gather)
            t2_gathers = tf.gather(t4_10nn_strs, nn_id)
            t2_one_nns = tf.as_string(t2_gathers)
            postfix = underscore + t2_one_nns + filetype
            fname = get_filename(t2_one_nn, postfix)
            t4_10nn_fnames = fname if nn_id == 0 else tf.concat(axis=0, values=[t4_10nn_fnames, fname])


        # [('000000000927_1.jpg', 0.03125), ('000000568135_2.jpg', 19095.953), ('000000187857_1.jpg', 23359.39),
        #  ('000000521998_2.jpg', 23557.688), ('000000140816_1.jpg', 24226.852), ('000000015109_1.jpg', 25191.469),
        #  ('000000525567_1.jpg', 25484.93), ('000000377422_1.jpg', 25654.125), ('000000269815_2.jpg', 26794.836),
        #  ('000000345617_2.jpg', 26872.812)]

        ########################################################################################################

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        max = 5
        cnt = 0

        try:
            while not coord.should_stop():
                print('t1_10nn_str type: %s' % type(t1_10nn_str))
                fns, t_imgs, t1_fns, t2_fns, t3_fns, t4_fns = sess.run([filename, train_images, t1_10nn_fnames, t2_10nn_fnames, t3_10nn_fnames, t4_10nn_fnames])

                print('fns.shape: %s' % str(fns.shape))
                print('t1_fns.shape: %s' % str(t1_fns.shape))

                for i in range(batch_size):
                    print('ITERATION [%d] >>>>>>' % i)
                    fname = fns[i].decode("utf-8")
                    t_img = t_imgs[i]
                    name = os.path.join(filedir_out, fname)
                    print('save I_ref to %s...' % name)
                    imsave(name, t_img)

                    f_o = os.path.join(tile_filedir_out, 'I_ref_' + fname)
                    print('cp %s %s' % (name, f_o))

                    t1_10nn = [e.decode("utf-8") for e in t1_fns[i]]
                    t2_10nn = [e.decode("utf-8") for e in t2_fns[i]]
                    t3_10nn = [e.decode("utf-8") for e in t3_fns[i]]
                    t4_10nn = [e.decode("utf-8") for e in t4_fns[i]]

                    print('I_ref: %s' % fname)
                    print('t1 10-NN:')  #  % str(t1_10nn))
                    for j in range(10):
                        t_f = os.path.join(tile_filedir_in, 't1')
                        t_f = os.path.join(t_f, t1_10nn[j])
                        t_o = os.path.join(tile_filedir_out, 't1', str(j+1) + '_' + t1_10nn[j])
                        print('cp %s %s' % (t_f, t_o))
                    print('-----')
                    print('t2 10-NN:') # %s' % str(t2_10nn))
                    for j in range(10):
                        t_f = os.path.join(tile_filedir_in, 't2')
                        t_f = os.path.join(t_f, t2_10nn[j])
                        t_o = os.path.join(tile_filedir_out, 't2', str(j+1) + '_' + t2_10nn[j])
                        print('cp %s %s' % (t_f, t_o))
                    print('-----')
                    print('t3 10-NN:') # %s' % str(t3_10nn))
                    for j in range(10):
                        t_f = os.path.join(tile_filedir_in, 't3')
                        t_f = os.path.join(t_f, t3_10nn[j])
                        t_o = os.path.join(tile_filedir_out, 't3', str(j+1) + '_' + t3_10nn[j])
                        print('cp %s %s' % (t_f, t_o))
                    print('-----')
                    print('t4 10-NN:') # %s' % str(t4_10nn))
                    for j in range(10):
                        t_f = os.path.join(tile_filedir_in, 't4')
                        t_f = os.path.join(t_f, t4_10nn[j])
                        t_o = os.path.join(tile_filedir_out, 't4', str(j+1) + '_' + t4_10nn[j])
                        print('cp %s %s' % (t_f, t_o))
                    print('-----')

                    print('ITERATION [%d] <<<<<<' % i)

                cnt = cnt + 1
                if cnt >= max:
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
                'image/knn/t2': tf.VarLenFeature(tf.int64),
                'image/knn/t2s': tf.VarLenFeature(tf.int64),
                'image/knn/t3': tf.VarLenFeature(tf.int64),
                'image/knn/t3s': tf.VarLenFeature(tf.int64),
                'image/knn/t4': tf.VarLenFeature(tf.int64),
                'image/knn/t4s': tf.VarLenFeature(tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)
    filename = features['image/filename']

    t1_10nn_str = features['image/knn/t1']
    t1_10nn_strs = features['image/knn/t1s']
    t2_10nn_str = features['image/knn/t2']
    t2_10nn_strs = features['image/knn/t2s']
    t3_10nn_str = features['image/knn/t3']
    t3_10nn_strs = features['image/knn/t3s']
    t4_10nn_str = features['image/knn/t4']
    t4_10nn_strs = features['image/knn/t4s']

    orig_image = features['image/encoded']

    image = preprocess_image(orig_image, img_size, img_w, img_h)

    return filename, image, t1_10nn_str, t2_10nn_str, t3_10nn_str, t4_10nn_str, t1_10nn_strs, t2_10nn_strs, t3_10nn_strs, t4_10nn_strs


def preprocess_image(orig_image, img_size, img_w, img_h):
    oi1 = tf.image.decode_jpeg(orig_image)
    size = tf.minimum(img_h, img_w)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape)
    image = tf.image.resize_images(image, [img_size, img_size])
    image = tf.reshape(image, (img_size, img_size, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1
    return image


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)


