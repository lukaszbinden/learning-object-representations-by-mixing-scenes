import tensorflow as tf

# LZ 15.04:
# adapted from https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
def apply_noise(x, noise_var=None, randomize_noise=True):
    '''
    In short: creates a single-channel image consisting of uncorrelated Gaussian noise, applies learned per-channel
    scaling factors to the noise input and adds it to the input x.

    From the paper: we provide our generator with a direct means to generate stochastic detail by introducing explicit noise
    inputs. These are single-channel images consisting of uncorrelated Gaussian noise, and we feed a dedicated noise
    image to each layer of the synthesis network. The noise image is broadcasted to all feature maps using learned per-feature
    scaling factors and then added to the output of the corresponding convolution.

    :param x:
    :param noise_var:
    :param randomize_noise:
    :return:
    '''
    assert len(x.shape) == 4 # NCHW
    with tf.variable_scope('Noise'):
        if noise_var is None or randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1], dtype=x.dtype)
        else:
            noise = tf.cast(noise_var, x.dtype)
        weight = tf.get_variable('weight', shape=[x.shape[3].value], initializer=tf.initializers.zeros())
        print("created %d noise weights..." % x.shape[3].value)
        return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, 1, 1, -1])