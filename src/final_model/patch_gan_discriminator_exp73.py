from ops_alex_exp73 import *
from ops_coordconv_exp73 import *

class Deep_PatchGAN_Discrminator(object):

    def __init__(self, hidden_activation=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm, addCoordConv=False, isTraining=True, flags=None):
        self.hidden_activation = hidden_activation
        self.flags = flags
        self.normalizer_fn = normalizer_fn
        self.addCoordConv = addCoordConv
        self.isTraining = isTraining

    def __call__(self, x, **kwargs):
        df_dim = 42

        if self.addCoordConv:
            h0 = tf.nn.leaky_relu(conv2d(coord_conv(x), df_dim, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=False, name='d_1_h0_conv'))       # 42

            h1 = tf.nn.leaky_relu(conv2d(coord_conv(h0), df_dim * 2, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_1_h1_conv'))   # 84

            h2 = tf.nn.leaky_relu(conv2d(coord_conv(h1), df_dim * 4, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_1_h2_conv'))   # 168

            h3 = tf.nn.leaky_relu(conv2d(coord_conv(h2), df_dim * 8, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_1_h3_conv'))   # 336

            h4 = h3
            # h4 = tf.nn.leaky_relu(conv2d(coord_conv(h3), df_dim * 8, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_1_h4_conv'))   # 336

        else:
            h0 = tf.nn.leaky_relu(conv2d(x, df_dim  * 1, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=False, name='d_1_h0_conv'))

            h1 = tf.nn.leaky_relu(conv2d(h0, df_dim * 2, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_1_h1_conv'))

            h2 = tf.nn.leaky_relu(conv2d(h1, df_dim * 4, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_1_h2_conv'))

            h3 = tf.nn.leaky_relu(conv2d(h2, df_dim * 8, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_1_h3_conv'))

            h4 = h3
            # h4 = tf.nn.leaky_relu(conv2d(h3, df_dim * 8, k_h=4, k_w=4, d_h=2, d_w=2, use_spectral_norm=True, name='d_1_h4_conv'))


        h5 = conv2d(h4, 1, k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=False, name='d_1_h5_conv')

        return h5

        # with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as _:
        # h0 = tf.contrib.layers.conv2d(inputs=x,
        #                               activation_fn=self.hidden_activation,
        #                               num_outputs=df_dim,
        #                               kernel_size=4,
        #                               stride=2,
        #                               normalizer_fn=None)
        #
        # h1 = tf.contrib.layers.conv2d(inputs=h0,
        #                               activation_fn=self.hidden_activation,
        #                               num_outputs=df_dim * 2,
        #                               kernel_size=4,
        #                               stride=2,
        #                               normalizer_fn=self.normalizer_fn,
        #                               normalizer_params={'is_training': self.isTraining})
        #
        # h2 = tf.contrib.layers.conv2d(inputs=h1,
        #                               activation_fn=self.hidden_activation,
        #                               num_outputs=df_dim * 4,
        #                               kernel_size=4,
        #                               stride=2,
        #                               normalizer_fn=self.normalizer_fn,
        #                               normalizer_params={'is_training': self.isTraining})
        #
        # h3 = tf.contrib.layers.conv2d(inputs=h2,
        #                               activation_fn=self.hidden_activation,
        #                               num_outputs=df_dim * 8,
        #                               kernel_size=4,
        #                               stride=2,
        #                               normalizer_fn=self.normalizer_fn,
        #                               normalizer_params={'is_training': self.isTraining})
        #
        # h4 = tf.contrib.layers.conv2d(inputs=h3,
        #                               activation_fn=self.hidden_activation,
        #                               num_outputs=df_dim * 16,
        #                               kernel_size=4,
        #                               stride=2,
        #                               normalizer_fn=self.normalizer_fn,
        #                               normalizer_params={'is_training': self.isTraining})
        #
        # h5 = tf.contrib.layers.conv2d(inputs=h4,
        #                               activation_fn=None,
        #                               num_outputs=1,
        #                               kernel_size=1,
        #                               stride=1,
        #                               normalizer_fn=None)


        """
        # average pooling h4 from (8 x 8 x 1) to 1
        h5 = tf.layers.average_pooling2d(inputs=h4, pool_size=h4.get_shape()[1:3], strides=h4.get_shape()[1:3])

        output = patch_gan_linear(tf.reshape(h5, [self.flags.blur_batch_size, -1]), 1, scope=scope,
                                  batch_size=self.flags.blur_batch_size)
        """

        #return tf.nn.sigmoid(h5), h5

