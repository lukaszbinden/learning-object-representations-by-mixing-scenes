"""
Michele Alberti:
For reference, I used this code to debug yours :)
See https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
for a very nice visualization (is not the accepted answer you need to scroll a little)
"""
import tensorflow as tf
import numpy as np

def test_conv2d_transpose():
    # input batch shape = (1, 2, 2, 1) -> (batch_size, height, width, channels) - 2x2x1 image in batch of 1
    x = tf.constant(np.array([[
        [[1], [2]],
        [[3], [4]]
    ]]), tf.float32)

    # shape = (3, 3, 1, 1) -> (height, width, input_channels, output_channels) - 3x3x1 filter
    f = tf.constant(np.array([
        [[[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]]]
    ]), tf.float32)

    conv = tf.nn.conv2d_transpose(x, f, output_shape=(1, 4, 4, 1), strides=[1, 2, 2, 1], padding='SAME')

    with tf.Session() as session:
        result = session.run(conv)

    assert (np.array([[
        [[1.0], [1.0],  [3.0], [2.0]],
        [[1.0], [1.0],  [3.0], [2.0]],
        [[4.0], [4.0], [10.0], [6.0]],
        [[3.0], [3.0],  [7.0], [4.0]]]]) == result).all()
    print('verified.')


test_conv2d_transpose()
