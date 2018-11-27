# source 1: https://github.com/taki0112/Densenet-Tensorflow
# source 2 (upsampling): https://github.com/GeorgeSeif/Semantic-Segmentation-Suite

from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from ops_alex import *

# Hyperparameter
# growth_k = 24
# nb_block = 2 # how many (dense block + Transition Layer) ?
# init_learning_rate = 1e-4
# epsilon = 1e-4 # AdamOptimizer epsilon
# dropout_rate = 0.2
#
# # Momentum Optimizer will use
# nesterov_momentum = 0.9
# weight_decay = 1e-4
#
# # Label & batch_size
# batch_size = 64
#
# iteration = 782
# # batch_size * iteration = data_set_number
#
# test_iteration = 10
#
# total_epochs = 300

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def conv_transpose_layer(input, filters_keep, kernel=[3,3], stride=[2,2], layer_name="conv_transp"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d_transpose(inputs=input, filters=filters_keep, kernel_size=kernel, strides=stride, padding='SAME', use_bias=False)
        return network

def Global_Average_Pooling(x):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def instance_normalization(x):
    return instance_norm(x)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x, output_dim, name_prefix) :
    return tf.layers.dense(inputs=x, units=output_dim, name=name_prefix + 'linear')

# def Evaluate(sess):
#     test_acc = 0.0
#     test_loss = 0.0
#     test_pre_index = 0
#     add = 1000
#
#     for it in range(test_iteration):
#         test_batch_x = test_x[test_pre_index: test_pre_index + add]
#         test_batch_y = test_y[test_pre_index: test_pre_index + add]
#         test_pre_index = test_pre_index + add
#
#         test_feed_dict = {
#             x: test_batch_x,
#             label: test_batch_y,
#             learning_rate: epoch_learning_rate,
#             training_flag: False
#         }
#
#         loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)
#
#         test_loss += loss_ / 10.0
#         test_acc += acc_ / 10.0
#
#     summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
#                                 tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
#
#     return test_acc, test_loss, summary

class DenseNetEncoder:
    def __init__(self, x, filters, dropout_rate, training, output_dim):
        self.nb_blocks = None # nb_blocks currently not used
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.training = training
        self.output_dim = output_dim
        self.name_prefix = 'g_1_'
        self.model = self.encoder_dense_net(x)


    def bottleneck_layer(self, x, scope):
        # this type of bottleneck layer is refered to in the paper
        # as DenseNet-B
        # print(x)
        with tf.name_scope(scope):
            x = instance_normalization(x)
            x = Relu(x)
            # rationale behind 1x1 conv:
            # reduce the number of input feature-maps to 4*filters, and thus improve
            # computational efficiency
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            x = instance_normalization(x)
            x = Relu(x)
            # produce 'filters' new feature map to concatenate to the "global state" i.e.
            # the feature maps from the previous layers
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_down_layer(self, x, scope):
        # rationale: To further improve model compactness,
        # we can reduce the number of feature-maps at transition layers.
        # Here theta = 0.5 because of pooling layer reducing the spatial dimension by
        # factor of 2. This is called DenseNet-BC.
        with tf.name_scope(scope):
            x = instance_normalization(x)
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def encoder_dense_net(self, input_x):
        # first conv layer according to ImageNet experiment in paper
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name=self.name_prefix + 'conv0')

        #x = conv2d(input_x, df_dim, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv0')

        # x = Max_Pooling(x, pool_size=[3,3], stride=2)


        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_down_layer(x, scope='trans_'+str(i))
        """

        # paper: except for ImageNet, the DenseNet used in our experiments has three
        # dense blocks that each has an equal number of layers. Former see table 1.

        x = self.dense_block(input_x=x, nb_layers=6, layer_name=self.name_prefix + 'dense_1')
        x = self.transition_down_layer(x, scope=self.name_prefix + 'trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_down_layer(x, scope='trans_2')

        #
        # x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        # x = self.transition_down_layer(x, scope='trans_3')

        # -> receptive field: 128x
        x = self.dense_block(input_x=x, nb_layers=8, layer_name=self.name_prefix + 'dense_final')

        # 100 Layer
        # x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = instance_normalization(x)
        x = Relu(x)
        x = Global_Average_Pooling(x)

        # x = flatten(x)
        # x = Linear(x, self.output_dim, self.name_prefix)

        return x


class DenseNetDecoder:
    def __init__(self, x, filters, dropout_rate, training):
        self.nb_blocks = None # nb_blocks currently not used
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.training = training
        self.name_prefix = 'g_dec_'
        self.model = self.decoder_densenet(x)


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = instance_normalization(x)
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            x = instance_normalization(x)
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            return x

    def transition_up_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_transpose_layer(x, filters_keep=self.filters, layer_name=scope+'_conv_transp1')
            x = instance_normalization(x)
            x = Relu(x)
            # x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            # x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def decoder_densenet(self, input_x):

        x = self.transition_up_layer(input_x, scope=self.name_prefix + 'trans_up_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name=self.name_prefix + 'dense_1')

        x = self.transition_up_layer(x, scope=self.name_prefix + 'trans_up_2')

        x = self.dense_block(input_x=x, nb_layers=6, layer_name=self.name_prefix + 'dense_2')

        print('output before last layer:', str(x.shape))

        x = conv_transpose_layer(x, filters_keep=3, kernel=[1,1], stride=[1,1], layer_name=self.name_prefix + '_conv_transp2')

        print('output of dense_net_decoder is:')
        print(x)
        assert 1 == 2

        return tf.nn.tanh(x)

        # x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name=self.name_prefix + 'conv0')
        #
        # #x = conv2d(input_x, df_dim, k_h=4, k_w=4, use_spectral_norm=True, name='g_1_conv0')
        #
        # # x = Max_Pooling(x, pool_size=[3,3], stride=2)
        #
        #
        # """
        # for i in range(self.nb_blocks) :
        #     # 6 -> 12 -> 48
        #     x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
        #     x = self.transition_down_layer(x, scope='trans_'+str(i))
        # """
        #
        #
        # x = self.dense_block(input_x=x, nb_layers=12, layer_name=self.name_prefix + 'dense_1')
        # x = self.transition_up_layer(x, scope=self.name_prefix + 'trans_1')
        #
        # # x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        # # x = self.transition_down_layer(x, scope='trans_2')
        # #
        # # x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        # # x = self.transition_down_layer(x, scope='trans_3')
        #
        # x = self.dense_block(input_x=x, nb_layers=12, layer_name=self.name_prefix + 'dense_final')
        #
        # # 100 Layer
        # # x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        # x = instance_normalization(x)
        # x = Relu(x)
        # x = Global_Average_Pooling(x)
        # x = flatten(x)
        # x = Linear(x, self.output_dim, self.name_prefix)

        # return x


def encoder_dense(tile_image, batch_size, feature_size, is_train=True, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # Hyperparameter --->>
    growth_k = 12
    dr = 0.2
    # Hyperparameter ---<<

    logits = DenseNetEncoder(x=tile_image, filters=growth_k,
                             dropout_rate=dr, training=is_train,
                             output_dim=feature_size).model

    print('output of encoder_dense is:', str(logits.shape))

    assert logits.shape[0] == batch_size
    assert logits.shape[1] == feature_size

    return logits


def decoder_dense(representations, batch_size, is_train=True, reuse=False):

    # TODO: growth rate => shrink rate (i.e. is decreasing)

    if reuse:
        tf.get_variable_scope().reuse_variables()

    # Hyperparameter --->>
    growth_k = 24
    dr = 0.2
    # Hyperparameter ---<<

    images = DenseNetDecoder(x=representations, filters=growth_k,
                             dropout_rate=dr, training=is_train).model

    print('output of encoder_dense is:', str(images.shape))

    assert images.shape[0] == batch_size
    assert images.shape[1] == 128
    assert images.shape[2] == 128
    assert images.shape[3] == 3

    return images
