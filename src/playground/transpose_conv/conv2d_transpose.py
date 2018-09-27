import tensorflow as tf
import numpy as np

input = tf.placeholder(dtype=tf.float32, shape=(1, 4, 4, 1))

t = tf.get_variable('t', (3,3,1,1),tf.float32)
w = tf.ones_like(t)

# TODO 17.9: how does output (1, 8, 8, 1) work with padding='SAME' ?
conv = tf.nn.conv2d_transpose(input, w, output_shape=(1, 7, 7, 1), strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #xin = np.random.rand(1,4,4,1)
    #print(xin)
    x = [[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]]
    b = np.array(x)
    b = b.reshape((1,4,4,1))
    out = sess.run(conv, feed_dict={input:b})
    print(out.shape)
    #print(out)
    print('----')
    print(out.reshape((7,7)))



