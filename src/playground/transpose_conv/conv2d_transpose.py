import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.ops import init_ops

input = tf.placeholder(dtype=tf.float32, shape=(1, 2, 2, 1))

t = tf.get_variable('t', (3,3,1,1), tf.float32)
w = tf.ones_like(t)

# TODO 17.9: how does output (1, 8, 8, 1) work with padding='SAME' ?
conv = tf.nn.conv2d_transpose(input, w, output_shape=(1, 5, 5, 1), strides=[1, 2, 2, 1], padding='VALID')

conv2 = slim.conv2d_transpose(input, 1, kernel_size=[3, 3], stride=[2, 2], padding='VALID', weights_initializer=init_ops.ones_initializer(), activation_fn=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #xin = np.random.rand(1,4,4,1)
    #print(xin)
    #x = [[2,2],[3,3],[4,4],[5,5]]
    x = [2, 3, 4, 5]
    b = np.array(x)
    b = b.reshape((1,2,2,1))
    out2 = sess.run(conv2, feed_dict={input: b})
    print(out2.shape)
    print('----')
    print(out2.reshape((5,5)))

    out = sess.run(conv, feed_dict={input:b})
    print(out.shape)
    #print(out)
    print('----')
    print(out.reshape((5,5)))



