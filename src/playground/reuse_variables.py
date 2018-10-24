import tensorflow as tf
import sys


def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def main(argv):
    input1 = tf.random_normal([1,10,10,32])
    input2 = tf.random_normal([1,20,20,32])
    with tf.variable_scope("conv1") as conv1:
        x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
        with tf.variable_scope("attention", reuse=False):
            x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])

        conv1.reuse_variables()

        with tf.variable_scope("attention", reuse=False):
            x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])

    # with tf.variable_scope(conv1, reuse=True):
        #conv1.reuse_variables()
        #conv1.reuse_variables()
    #    x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])


if __name__ == '__main__':
    tf.app.run(argv=sys.argv)
