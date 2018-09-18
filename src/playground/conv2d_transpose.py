import tensorflow as tf
import numpy as np

#dcout = tf.layers.conv2d_transpose(x, 64, 4, 3, padding="valid")
input = tf.placeholder(dtype=tf.float32, shape=(1, 5, 5, 1))

t = tf.get_variable('t', (3,3,1,1),tf.float32)
w = tf.ones_like(t)

# TODO 17.9: how does output (1, 8, 8, 1) or (1, 7, 7, 1) work with padding='SAME' ?
conv = tf.nn.conv2d_transpose(input, w, output_shape=(1, 8, 8, 1), strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xin = np.random.rand(1,5,5,1)
    out = sess.run(conv, feed_dict={input:xin})
    print(out.shape)



# import tensorflow as tf
# import numpy as np
#
# def test_conv2d_transpose():
#     # input batch shape = (1, 4, 4, 1) -> (batch_size, height, width, channels) - 2x2x1 image in batch of 1
#     input = tf.constant(np.array([[
#         [[1], [2]],
#         [[3], [4]]
#     ]]), tf.float32)
#
#     # shape = (3, 3, 1, 1) -> (height, width, input_channels, output_channels) - 3x3x1 filter
#     w = tf.constant(np.array([
#         [[[1]], [[1]], [[1]]],
#         [[[1]], [[1]], [[1]]],
#         [[[1]], [[1]], [[1]]]
#     ]), tf.float32)
#
#     conv = tf.nn.conv2d_transpose(input, w, output_shape=(1, 8, 8, 1), strides=[1, 2, 2, 1], padding='SAME')
#
#     with tf.Session() as session:
#         result = session.run(conv)
#
#     assert (np.array([[
#         [[1.0], [1.0],  [3.0], [2.0]],
#         [[1.0], [1.0],  [3.0], [2.0]],
#         [[4.0], [4.0], [10.0], [6.0]],
#         [[3.0], [3.0],  [7.0], [4.0]]]]) == result).all()
