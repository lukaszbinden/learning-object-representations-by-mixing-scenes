import tensorflow as tf
import glob
import sys
import numpy as np
import scipy
from ops_alex import *
from utils_dcgan import *
from utils_common import *

def main(_):
    with tf.Session() as sess:
        epochs = 1
        batch_size = 5
        reader = tf.TFRecordReader()
        read_fn = lambda name : read_record(name, reader)
        h, w, crop_shape, train_images = get_pipeline('2017_train_small_anys.tfrecords', batch_size, epochs, read_fn)
        read_fn_sc_def = lambda name, sc : read_record_2(name, reader, sc)
        read_fn_sc = lambda name : read_fn_sc_def(name, 9)
        h2, w2, crop_shape2, train_images_2 = get_pipeline('2017_train_small_anys.tfrecords', batch_size, epochs, read_fn_sc)
        read_fn_sc = lambda name : read_fn_sc_def(name, 8)
        h3, w3, crop_shape3, train_images_3 = get_pipeline('2017_train_small_anys.tfrecords', batch_size, epochs, read_fn_sc)
        read_fn_sc = lambda name : read_fn_sc_def(name, 7)
        h4, w4, crop_shape4, train_images_4 = get_pipeline('2017_train_small_anys.tfrecords', batch_size, epochs, read_fn_sc)
        read_fn_sc = lambda name : read_fn_sc_def(name, 6)
        h5, w5, crop_shape5, train_images_5 = get_pipeline('2017_train_small_anys.tfrecords', batch_size, epochs, read_fn_sc)
        # read_fn_sc = lambda name : read_fn_sc_def(name, 5)
        # h6, w6, crop_shape6, train_images_6 = get_pipeline('2017_train_small_anys.tfrecords', batch_size, epochs, read_fn_sc)
        # read_fn_sc = lambda name : read_fn_sc_def(name, 4)
        # h7, w7, crop_shape7, train_images_7 = get_pipeline('2017_train_small_anys.tfrecords', batch_size, epochs, read_fn_sc)
        # read_fn_sc = lambda name : read_fn_sc_def(name, 3)
        # h8, w8, crop_shape8, train_images_8 = get_pipeline('2017_train_small_anys.tfrecords', batch_size, epochs, read_fn_sc)

        # rec_loss_f1_f2 = tf.reduce_mean(tf.square(train_images - train_images_2))
        # rec_loss_f1_f3 = tf.reduce_mean(tf.square(train_images - train_images_3))
        # rec_loss_f1_f4 = tf.reduce_mean(tf.square(train_images - train_images_4))
        # rec_loss_f1_f5 = tf.reduce_mean(tf.square(train_images - train_images_5))


        I1_f_4 = encoder(train_images, batch_size)
        tf.get_variable_scope().reuse_variables()
        i2_f_4 = encoder(train_images_2, batch_size)
        i3_f_4 = encoder(train_images_3, batch_size)
        i4_f_4 = encoder(train_images_4, batch_size)
        i5_f_4 = encoder(train_images_5, batch_size)

        rec_loss_f4_f2 = tf.reduce_mean(tf.square(I1_f_4 - i2_f_4), 1)
        rec_loss_f4_f3 = tf.reduce_mean(tf.square(I1_f_4 - i3_f_4), 1)
        rec_loss_f4_f4 = tf.reduce_mean(tf.square(I1_f_4 - i4_f_4), 1)
        rec_loss_f4_f5 = tf.reduce_mean(tf.square(I1_f_4 - i5_f_4), 1)

        all = tf.stack(axis=0, values=[rec_loss_f4_f2, rec_loss_f4_f3, rec_loss_f4_f4, rec_loss_f4_f5])
        print(all)
        assert all.shape[0] == 4
        assert all.shape[1] == batch_size
        ind = tf.argmin(all, axis=0)
        print(ind)
        # assert ind.shape[0] == batch_size
        all_tile4 = tf.concat([train_images_2, train_images_3, train_images_4, train_images_5], axis=0)
        all_f4 = tf.concat([i2_f_4, i3_f_4, i4_f_4, i5_f_4], axis=0)
        # print('all_f4: ', all_f4.shape)
        for i in range(batch_size):
            # choose which batch i holds the L2-closest tile (i.e. w/ lowest L2 loss), then choose that tile at position i
            tile = all_tile4[ind[i] * batch_size:(ind[i] + 1) * batch_size][i]
            tile = tf.expand_dims(tile, 0)
            # print('tile: ', tile.shape)
            J_4_tile = tile if i == 0 else tf.concat(axis=0, values=[J_4_tile, tile])

            f = all_f4[ind[i] * batch_size:(ind[i] + 1) * batch_size][i]
            f = tf.expand_dims(f, 0)
            # print('f: ', f.shape)
            J_4_f = f if i == 0 else tf.concat(axis=0, values=[J_4_f, f])
            # print('J_4_f: ', J_4_f.shape)




        # #square = tf.sum(tf.square(train_images - train_images_2))
        # kd = tf.reduce_mean(tf.square(train_images - train_images_2), 1)
        # # print(kd.shape)
        # kd = tf.reduce_mean(tf.square(train_images - train_images_2), 1, keep_dims=True)
        # # print(kd.shape)
        # kr = tf.reduce_mean(tf.square(train_images - train_images_2), [1,2,3])
        #
        # all_img = tf.concat([train_images_2, train_images_3, train_images_4, train_images_5], axis=0)
        # # print(all_img.shape)
        # all = tf.stack([rec_loss_f1_f2, rec_loss_f1_f3, rec_loss_f1_f4, rec_loss_f1_f5])
        # argmin = tf.argmin(all, axis=0)
        #
        # final_img = all_img[argmin * batch_size:(argmin + 1) * batch_size]
        #
        # s1 = tf.reduce_sum(tf.square(train_images) )
        # s2 = tf.reduce_sum(tf.square(train_images_2))
        # s3 = tf.reduce_sum(tf.square(train_images_3))
        # s4 = tf.reduce_sum(tf.square(train_images_4))
        # s5 = tf.reduce_sum(tf.square(train_images_5))
        # s6 = tf.reduce_sum(tf.square(train_images_6))
        # s7 = tf.reduce_sum(tf.square(train_images_7))
        # s8 = tf.reduce_sum(tf.square(train_images_8))
        # f = tf.reduce_sum( tf.square(final_img     ))

        # imgs_per_group = int(2/2)
        # group = lambda i: train_images[(i * imgs_per_group):imgs_per_group + (i * imgs_per_group), :, :, :]
        # images_i1 = group(0)
        # images_i2 = group(1)

        # for _ in range(3):
        #     print('images_i1.shape[1]: ' + images_i1.shape[1])
        #     print('images_i1.shape[2]: ' + images_i1.shape[2])
        #     size = tf.minimum(images_i1.shape[1], images_i1.shape[2])
        #     print('i1: %s -> %s' % (images_i1.shape, size.shape))
        #     crop_shape = tf.parallel_stack([size, size, 3])
        #     crop = tf.random_crop(images_i1, crop_shape)
        #     print(type(crop))
        #     print(crop.shape)

        ######################################################################3

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        try:
            while not coord.should_stop():
                #hr, wr, i1, i2 = sess.run([h, w, images_i1, images_i2])
                # rs1,rs2,rs3,rs4,rs5,rs6,rs7,rs8,rf \
                #     , am \
                #     , rec_loss_f1_f2_, rec_loss_f1_f3_, rec_loss_f1_f4_, rec_loss_f1_f5_ \
                #     , tr, tr_2, tr_3, tr_4, tr_5, fi = sess.run([
                #                                              s1,s2,s3,s4,s5,s6,s7,s8,f \
                #                                              , argmin \
                #                                              , rec_loss_f1_f2, rec_loss_f1_f3, rec_loss_f1_f4, rec_loss_f1_f5 \
                #                                              , train_images, train_images_2, train_images_3, train_images_4, train_images_5, final_img])



                # ind,r1, r2, r3, r4,r5,f= sess.run([ind,train_images, train_images_2,train_images_3,train_images_4,train_images_5, J_4_tile])
                #
                # print(ind)
                # print('save 1')
                # for i in range(batch_size):
                #     fn = 'train_images_%d.jpeg' % i
                #     scipy.misc.imsave(fn, r1[i])
                # print('save 2')
                # for i in range(batch_size):
                #     scipy.misc.imsave('train_images_2_%d.jpeg' % i, r2[i])
                # print('save 3')
                # for i in range(batch_size):
                #     scipy.misc.imsave('train_images_3_%d.jpeg' % i, r3[i])
                # print('save 4')
                # for i in range(batch_size):
                #     scipy.misc.imsave('train_images_4_%d.jpeg' % i, r4[i])
                # print('save 5')
                # for i in range(batch_size):
                #     scipy.misc.imsave('train_images_5_%d.jpeg' % i, r5[i])
                # print('save 6')
                # print(f.shape)
                # for i in range(batch_size):
                #     scipy.misc.imsave('J_4_tile_%d.jpeg' % i, f[i])

                print('--------------------------------------')
                ind,l1,l2,l3,l4,l5,l6= sess.run([ind,I1_f_4,i2_f_4, i3_f_4,i4_f_4,i5_f_4,J_4_f])

                print(ind)
                print('i1_f_4:>>>>>')
                print(str(np.linalg.norm(l1[0])))
                print(str(np.linalg.norm(l1[1])))
                print(str(np.linalg.norm(l1[2])))
                print(str(np.linalg.norm(l1[3])))
                print(str(np.linalg.norm(l1[4])))
                print('i1_f_4.<<<<<')
                print('i2_f_4:>>>>>')
                print(str(np.linalg.norm(l2[0])))
                print(str(np.linalg.norm(l2[1])))
                print(str(np.linalg.norm(l2[2])))
                print(str(np.linalg.norm(l2[3])))
                print(str(np.linalg.norm(l2[4])))
                print('i2_f_4.<<<<<')
                print('i3_f_4:>>>>>')
                print(str(np.linalg.norm(l3[0])))
                print(str(np.linalg.norm(l3[1])))
                print(str(np.linalg.norm(l3[2])))
                print(str(np.linalg.norm(l3[3])))
                print(str(np.linalg.norm(l3[4])))
                print('i3_f_4.<<<<<')
                print('i4_f_4:>>>>>')
                print(str(np.linalg.norm(l4[0])))
                print(str(np.linalg.norm(l4[1])))
                print(str(np.linalg.norm(l4[2])))
                print(str(np.linalg.norm(l4[3])))
                print(str(np.linalg.norm(l4[4])))
                print('i4_f_4.<<<<<')
                print('i5_f_4:>>>>>')
                print(str(np.linalg.norm(l5[0])))
                print(str(np.linalg.norm(l5[1])))
                print(str(np.linalg.norm(l5[2])))
                print(str(np.linalg.norm(l5[3])))
                print(str(np.linalg.norm(l5[4])))
                print('i5_f_4.<<<<<')
                print('J_4_f:>>>>>')
                print(str(np.linalg.norm(l6[0])))
                print(str(np.linalg.norm(l6[1])))
                print(str(np.linalg.norm(l6[2])))
                print(str(np.linalg.norm(l6[3])))
                print(str(np.linalg.norm(l6[4])))
                print('J_4_f.<<<<<')
                # for i in range(batch_size):
                #     print(l6[i])

                break

                # print('22222222222222222222222222222222')
                # print('hr: %s' % str(hr))
                # print('wr: %s' % str(wr))
                # print('c:')
                # print(c)
                # print('22222222222222222222222222222222')
                # print('hr2: %s' % str(hr2))
                # print('wr2: %s' % str(wr2))
                # print('c2:')
                # print(c2)
                # print('22222222222222222222222222222222')
                # print('hr3: %s' % str(hr3))
                # print('wr3: %s' % str(wr3))
                # print('c3:')
                # print(c3)
                # print('22222222222222222222222222222222')
                # print('hr4: %s' % str(hr4))
                # print('wr4: %s' % str(wr4))
                # print('c4:')
                # print(c4)
                # print('22222222222222222222222222222222')
                # print('hr5: %s' % str(hr5))
                # print('wr5: %s' % str(wr5))
                # print('c5:')
                # print(c5)
                # print('22222222222222222222222222222222')
                # print('hr6: %s' % str(hr6))
                # print('wr6: %s' % str(wr6))
                # print('c6:')
                # print(c6)
                # print('22222222222222222222222222222222')
                # print('hr7: %s' % str(hr7))
                # print('wr7: %s' % str(wr7))
                # print('c7:')
                # print(c7)
                # print('22222222222222222222222222222222')
                # print('hr8: %s' % str(hr8))
                # print('wr8: %s' % str(wr8))
                # print('c8:')
                # print(c8)
                print('--------------------------------------------')
                # print('argmin: %s, %s, %s, %s, %s' % (str(am), str(rec_loss_f1_f2_), str(rec_loss_f1_f3_), str(rec_loss_f1_f4_), str(rec_loss_f1_f5_)))
                # print('img: %s, %s, %s, %s | %s %s' % (str(np.linalg.norm(tr_2 - tr)), str(np.linalg.norm(tr_3 - tr)),
                #                                    str(np.linalg.norm(tr_4 - tr)), str(np.linalg.norm(tr_5 - tr)),
                #                                    str(np.linalg.norm(fi - tr)), str(np.linalg.norm(fi - tr))))
                # print('sum: %d | %d %d %d %d %d %d %d | %d' %(rs1,rs2,rs3,rs4,rs5,rs6,rs7,rs8,rf))
                print('--------------------------------------------')

        except Exception as e:
            if hasattr(e, 'message') and  'is closed and has insufficient elements' in e.message:
                print('Done training -- epoch limit reached')
            else:
                print('Exception here, ending training..')
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(e)
                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def encoder(tile_image, batch_size):
        """
        returns: 1D vector f1 with size=self.feature_size
        """
        df_dim = 64

        s0 = lrelu(instance_norm(conv2d(tile_image, df_dim, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, df_dim * 2, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, df_dim * 4, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv2')))
        s3 = lrelu(instance_norm(conv2d(s2, df_dim * 8, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv3')))
        s4 = lrelu(instance_norm(conv2d(s3, df_dim * 8, k_h=2, k_w=2, use_spectral_norm=True, name='g_1_conv4')))
        rep = lrelu((linear(tf.reshape(s4, [batch_size, -1]), 256, 'g_1_fc')))

        return rep

def get_pipeline_q(filename_queue, batch_size, read_fn, read_threads=4):
    with tf.variable_scope('dump_reader'):
        with tf.device('/cpu:0'):
            example_list = [read_fn(filename_queue) for _ in range(read_threads)]

            return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                         capacity=100 + batch_size * 16,
                                         min_after_dequeue=100,
                                         enqueue_many=False)

def get_pipeline(dump_file, batch_size, epochs, read_fn, read_threads=4):
    with tf.variable_scope('dump_reader'):
        with tf.device('/cpu:0'):
            all_files = [dump_file]
            print('tfrecords: ' + str(all_files))
            filename_queue = tf.train.string_input_producer(all_files, num_epochs=epochs ,shuffle=True)
            #example_list = [read_record(filename_queue) for _ in range(read_threads)]
            example_list = [read_fn(filename_queue) for _ in range(read_threads)]

            return tf.train.shuffle_batch_join(example_list, batch_size=batch_size,
                                         capacity=100 + batch_size * 16,
                                         min_after_dequeue=100,
                                         enqueue_many=False)


def read_record(filename_queue, reader):
    # reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)

    orig_image = features['image/encoded']

    oi1 = tf.image.decode_jpeg(orig_image)
    size = tf.minimum(img_h, img_w)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape)
    image = tf.image.resize_images(image, [128, 128])
    image = tf.reshape(image, (128, 128, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return img_h, img_w, crop_shape, image


def read_record_2(filename_queue, reader, scale):
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string)})

    img_h = features['image/height']
    img_h = tf.cast(img_h, tf.int32)
    img_w = features['image/width']
    img_w = tf.cast(img_w, tf.int32)

    orig_image = features['image/encoded']

    oi1 = tf.image.decode_jpeg(orig_image)
    size = tf.minimum(img_h, img_w)
    size = tf.cast(tf.round(tf.divide(tf.multiply(size, scale), 10)), tf.int32)
    size = tf.maximum(size, 128)
    crop_shape = tf.parallel_stack([size, size, 3])
    image = tf.random_crop(oi1, crop_shape)
    image = tf.image.resize_images(image, [128, 128])
    image = tf.reshape(image, (128, 128, 3))
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    return img_h, img_w, crop_shape, image



if __name__ == '__main__':
    tf.app.run(argv=sys.argv)
