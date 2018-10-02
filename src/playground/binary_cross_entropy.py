import tensorflow as tf
tfd = tf.contrib.distributions
import numpy as np
from ops_alex import binary_cross_entropy_with_logits

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    # m = tfd.Bernoulli(0.9).sample(9)
    # r = sess.run(m)
    # print(r)

    m = np.array([1,1,1,0,1,0,1,0,1])
    mask = tf.constant(m, tf.float32)
    p = np.array([0.99, 0.99, 0.99, 0.01, 0.99, 0.01, 0.99, 0.01, 0.99])
    prediction = tf.constant(p, tf.float32)
    res = binary_cross_entropy_with_logits(mask, prediction)
    res = sess.run(res)
    print('cls_loss: %s' % str(res))

    p = np.array([0.1, 0.5, 0.99, 0.01, 0.99, 0.4, 0.99, 0.01, 0.99])
    prediction = tf.constant(p, tf.float32)
    res = binary_cross_entropy_with_logits(tf.ones_like(mask), prediction)
    res = sess.run(res)
    print('g_loss: %s' % str(res))

    res = binary_cross_entropy_with_logits(tf.zeros_like(mask), prediction)
    res = sess.run(res)
    print('dsc_loss_fake: %s' % str(res))

    m = np.array([1,1,1,1,1,1,1,1,1])
    mask = tf.constant(m, tf.float32)
    p = np.bitwise_xor(m, [1,1,1,1,1,1,1,1,1])
    #print(m)
    #print(p)
    prediction = tf.constant(p, tf.float32)
    res = binary_cross_entropy_with_logits(mask, prediction)
    res = sess.run(res)
    print('loss max: %s' % str(res))

    p = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
    images_x4 = tf.constant(p, tf.float32)
    #p = np.array([[[11,12,13],[14,15,16],[17,18,19]]])
    p = np.array([[[1.1,2.1,3.1],[4.1,5.1,6.1],[7.1,8.1,9.1]]])
    images_x1 = tf.constant(p, tf.float32)
    rec_loss_x4_x1 = tf.reduce_mean(tf.square(images_x4 - images_x1))
    rec_loss_x4_x1 = sess.run(rec_loss_x4_x1)
    print('rec_loss_x4_x1: %s' % rec_loss_x4_x1)
