import tensorflow as tf
import glob
import numpy as np
import sys
sys.path.append('..')
import os
from scipy.misc import imsave
import traceback
from datetime import datetime
from utils_dcgan import save_images

def main(_):
    with tf.Session() as sess:

        #tf.set_random_seed(4285)

        epochs = 1
        batch_size = 3  # must divide dataset size (some strange error occurs if not)
        image_size = 128

        tfrecords_file_in = '/data/cvg/lukas/datasets/coco/2017_training/tfrecords_l2mix_flip_tile_10-L2nn_4285/181115/'  # '../data/train-00011-of-00060.tfrecords'
        filedir_out_base = '../logs/prepare_tensors_for_CLS'
        # tile_filedir_in = '/data/cvg/lukas/datasets/coco/2017_training/clustering_224x224_4285/'
        # tile_filedir_out = '~/results/knn_results/'
        path_tile_base = tf.constant("/data/cvg/lukas/datasets/coco/2017_training/clustering_224x224_4285/")

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
        # t1 ############################################################################################
        path_prefix_t1 = path_tile_base + tf.constant("t1/")
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
        path_prefix_t2 = path_tile_base + tf.constant("t2/")
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
        path_prefix_t3 = path_tile_base + tf.constant("t3/")
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
        path_prefix_t4 = path_tile_base + tf.constant("t4/")
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


        # replace tile w/ max L2 wrt I_ref w/ respective tile of I_ref
        tau = 16000
        for id in range(batch_size):
            index = nn_id[id]

            t1_10nn_L2_b = tf.gather(t1_10nn_L2[id], index)
            t2_10nn_L2_b = tf.gather(t2_10nn_L2[id], index)
            t3_10nn_L2_b = tf.gather(t3_10nn_L2[id], index)
            t4_10nn_L2_b = tf.gather(t4_10nn_L2[id], index)
            all_L2 = tf.stack(axis=0, values=[t1_10nn_L2_b, t2_10nn_L2_b, t3_10nn_L2_b, t4_10nn_L2_b])
            argmax_L2 = tf.argmax(tf.reshape(all_L2, [-1]), axis=0)
            argmin_L2 = tf.argmin(tf.reshape(all_L2, [-1]), axis=0)

            # pick I_ref_t1 IFF t1 is argmax L2 or L2 > TAU and t1 is not argmin L2
            is_t1_maxL2 = tf.equal(argmax_L2, 0)
            is_t1_minL2 = tf.equal(argmin_L2, 0)
            cond_Iref_t1 = tf.logical_and(tf.logical_or(is_t1_maxL2, tf.greater(t1_10nn_L2_b, tau)), tf.logical_not(is_t1_minL2))

            cond_Iref_t1_s = tf.expand_dims(cond_Iref_t1, 0) if id == 0 else tf.concat(axis=0, values=[cond_Iref_t1_s, tf.expand_dims(cond_Iref_t1, 0)])

            tile_1 = tf.expand_dims(tf.where(cond_Iref_t1, I_ref_t1[id], t1_10nn_images[id]), 0)
            assignment_1 = tf.where(cond_Iref_t1, 0, 1)
            J_1_tile = tile_1 if id == 0 else tf.concat(axis=0, values=[J_1_tile, tile_1])


            is_t2_maxL2 = tf.equal(argmax_L2, 1)
            is_t2_minL2 = tf.equal(argmin_L2, 1)
            cond_Iref_t2 = tf.logical_and(tf.logical_or(is_t2_maxL2, tf.greater(t2_10nn_L2_b, tau)), tf.logical_not(is_t2_minL2))

            cond_Iref_t2_s = tf.expand_dims(cond_Iref_t2, 0) if id == 0 else tf.concat(axis=0, values=[cond_Iref_t2_s, tf.expand_dims(cond_Iref_t2, 0)])

            tile_2 = tf.expand_dims(tf.where(cond_Iref_t2, I_ref_t2[id], t2_10nn_images[id]), 0)
            assignment_2 = tf.where(cond_Iref_t2, 0, 1)
            J_2_tile = tile_2 if id == 0 else tf.concat(axis=0, values=[J_2_tile, tile_2])


            is_t3_maxL2 = tf.equal(argmax_L2, 2)
            is_t3_minL2 = tf.equal(argmin_L2, 2)
            cond_Iref_t3 = tf.logical_and(tf.logical_or(is_t3_maxL2, tf.greater(t3_10nn_L2_b, tau)), tf.logical_not(is_t3_minL2))

            cond_Iref_t3_s = tf.expand_dims(cond_Iref_t3, 0) if id == 0 else tf.concat(axis=0, values=[cond_Iref_t3_s, tf.expand_dims(cond_Iref_t3, 0)])

            tile_3 = tf.expand_dims(tf.where(cond_Iref_t3, I_ref_t3[id], t3_10nn_images[id]), 0)
            assignment_3 = tf.where(cond_Iref_t3, 0, 1)
            J_3_tile = tile_3 if id == 0 else tf.concat(axis=0, values=[J_3_tile, tile_3])


            is_t4_maxL2 = tf.equal(argmax_L2, 3)
            is_t4_minL2 = tf.equal(argmin_L2, 3)
            cond_Iref_t4 = tf.logical_and(tf.logical_or(is_t4_maxL2, tf.greater(t4_10nn_L2_b, tau)), tf.logical_not(is_t4_minL2))

            cond_Iref_t4_s = tf.expand_dims(cond_Iref_t4, 0) if id == 0 else tf.concat(axis=0, values=[cond_Iref_t4_s, tf.expand_dims(cond_Iref_t4, 0)])

            tile_4 = tf.expand_dims(tf.where(cond_Iref_t4, I_ref_t4[id], t4_10nn_images[id]), 0)
            assignment_4 = tf.where(cond_Iref_t4, 0, 1)
            J_4_tile = tile_4 if id == 0 else tf.concat(axis=0, values=[J_4_tile, tile_4])


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
        assert assignments_actual.shape[0] == batch_size
        assert assignments_actual.shape[1] == 4


        # [('000000000927_1.jpg', 0.03125), ('000000568135_2.jpg', 19095.953), ('000000187857_1.jpg', 23359.39),
        #  ('000000521998_2.jpg', 23557.688), ('000000140816_1.jpg', 24226.852), ('000000015109_1.jpg', 25191.469),
        #  ('000000525567_1.jpg', 25484.93), ('000000377422_1.jpg', 25654.125), ('000000269815_2.jpg', 26794.836),
        #  ('000000345617_2.jpg', 26872.812)]

        ########################################################################################################

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        max_batches = 1
        cnt_batches = 0

        max_iterations = batch_size
        timef = datetime.now().strftime('%Y%m%d_%H%M%S')
        filedir_out_base = os.path.join(filedir_out_base, timef)
        os.makedirs(filedir_out_base, exist_ok=True)

        try:
            while not coord.should_stop():

                # r, s = sess.run([t1_10nn_ids, t1_10nn_subids])
                # print(r)
                # print(s)

                print('assignments_actual.shape: ', assignments_actual.shape)

                img_ref, aa, inds, t1l2l, t2l2l, t3l2l, t4l2l, t1ids, t2ids, t3ids, t4ids, c1, c2, c3, c4, f1,f2,f3,f4,fr, \
                t1_img, t2_img, t3_img, t4_img, J1t,J2t,J3t,J4t,I1,I2,I3,I4 = sess.run([images_I_ref, assignments_actual, nn_id, t1_10nn_L2, t2_10nn_L2, t3_10nn_L2, t4_10nn_L2, \
                                                            t1_10nn_ids, t2_10nn_ids, t3_10nn_ids, t4_10nn_ids, \
                                                            cond_Iref_t1_s, cond_Iref_t2_s, cond_Iref_t3_s, cond_Iref_t4_s, \
                                                            t1_10nn_fnames, t2_10nn_fnames, t3_10nn_fnames, t4_10nn_fnames, filenames, \
                                                            t1_10nn_images, t2_10nn_images, t3_10nn_images, t4_10nn_images, \
                                                            J_1_tile, J_2_tile, J_3_tile, J_4_tile, \
                                                            I_ref_t1, I_ref_t2, I_ref_t3, I_ref_t4])

                cnt_iterations = 0
                for i in range(batch_size):
                    print('ITERATION [%d] >>>>>>' % i)

                    print('****************************************************************************************************************************************')
                    print('assignments_actual:')
                    print(aa[i])
                    print('index:')
                    print(inds[i])
                    print('t1_10nn_ids:')
                    print(t1ids[i])
                    print('t2_10nn_ids:')
                    print(t2ids[i])
                    print('t3_10nn_ids:')
                    print(t3ids[i])
                    print('t4_10nn_ids:')
                    print(t4ids[i])
                    print('t1_10nn_L2:')
                    print(t1l2l[i])
                    print('t2_10nn_L2:')
                    print(t2l2l[i])
                    print('t3_10nn_L2:')
                    print(t3l2l[i])
                    print('t4_10nn_L2:')
                    print(t4l2l[i])
                    print('t1_10nn_L2 selected:')
                    print(t1l2l[i][inds[i]])
                    print('t2_10nn_L2 selected:')
                    print(t2l2l[i][inds[i]])
                    print('t3_10nn_L2 selected:')
                    print(t3l2l[i][inds[i]])
                    print('t4_10nn_L2 selected:')
                    print(t4l2l[i][inds[i]])
                    print('condition: %s - %s - %s - %s' % (str(c1[i]), str(c2[i]), str(c3[i]), str(c4[i])))
                    print(fr[i].decode("utf-8"))
                    print(f1[i].decode("utf-8"))
                    print(f2[i].decode("utf-8"))
                    print(f3[i].decode("utf-8"))
                    print(f4[i].decode("utf-8"))
                    print('****************************************************************************************************************************************')

                    t_img = img_ref[i]
                    frn = fr[i].decode("utf-8")
                    name = os.path.join(filedir_out_base, ('%s_I_ref_' + frn) % i)
                    print('save I_ref to %s...' % name)
                    imsave(name, t_img)

                    # save_to_file(f1, filedir_out_base, i, t1_img)
                    # save_to_file(f2, filedir_out_base, i, t2_img)
                    # save_to_file(f3, filedir_out_base, i, t3_img)
                    # save_to_file(f4, filedir_out_base, i, t4_img)

                    grid_size = np.ceil(np.sqrt(batch_size))
                    grid = [grid_size, grid_size]

                    t_imgs = np.stack((t1_img[i], t2_img[i],t3_img[i],t4_img[i]))
                    assert t_imgs.shape[0] == 4
                    assert t_imgs.shape[1] == 64
                    assert t_imgs.shape[2] == 64
                    assert t_imgs.shape[3] == 3
                    save_images(t_imgs, grid, os.path.join(filedir_out_base, '%s_I_ref_t1-t4_%s.jpg' % (i, ''.join(str(e) for e in aa[i]))))
                    t_imgs = np.stack((J1t[i], J2t[i], J3t[i], J4t[i]))
                    assert t_imgs.shape[0] == 4
                    assert t_imgs.shape[1] == 64
                    assert t_imgs.shape[2] == 64
                    assert t_imgs.shape[3] == 3
                    save_images(t_imgs, grid, os.path.join(filedir_out_base, '%s_I_M_%s.jpg' % (i, ''.join(str(e) for e in aa[i]))))

                    print('variance:')
                    print(np.var(J1t[i]))
                    print(np.var(J2t[i]))
                    print(np.var(J3t[i]))
                    print(np.var(J4t[i]))


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


def save_to_file(f1, filedir_out_base, i, t1_img):
    fname = f1[i].decode("utf-8")
    fname = os.path.basename(fname)
    t_img = t1_img[i]
    name = os.path.join(filedir_out_base, ('%s_I_M_' + fname) % i)
    print('save I_M_ to %s...' % name)
    imsave(name, t_img)


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
        image = tf.cast(image, tf.float32) * (2. / 255) - 1
        with tf.control_dependencies([tf.assert_equal(batch_size, image.shape[0])]):
            return image
    else:
        image = tf.reshape(image, (img_size, img_size, 3))
        image = tf.cast(image, tf.float32) * (2. / 255) - 1
        return image


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)



