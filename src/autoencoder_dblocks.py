# LZ: Taken and adapted from:
# https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/models/FC_DenseNet_Tiramisu.py

from __future__ import division
import tensorflow.contrib.slim as slim
from ops_alex import *
from constants import *

def preact_conv(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.2):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """
    # preact = slim.batch_norm(inputs, fused=True)
    preact = instance_norm(inputs)
    preact = tf.nn.relu(preact)
    conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None,
                       weights_initializer=tf.random_normal_initializer(stddev=0.02, seed=4285), biases_initializer=tf.constant_initializer(0.01))
    if dropout_p != 0.0:
      conv = slim.dropout(conv, keep_prob=(1.0-dropout_p))
    return conv


def preact_conv_dec(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.2, name=None):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """

    # preact = slim.batch_norm(inputs, fused=True)
    preact = instance_norm(inputs)
    preact = tf.nn.relu(preact)

    #conv_slim = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None, biases_initializer=tf.constant_initializer(0.01))
    conv = conv2d(preact, n_filters, k_h=kernel_size[0], k_w=kernel_size[1], d_h=1, d_w=1, use_spectral_norm=True, name=name)

    if dropout_p != 0.0:
        conv = slim.dropout(conv, keep_prob=(1.0 - dropout_p))

    return conv


def DenseBlock(stack, n_layers, growth_rate, dropout_p, isDec=False, scope=None):
  """
  DenseBlock for DenseNet and FC-DenseNet
  Arguments:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
  with tf.name_scope(scope):
    new_features = []
    for i in range(n_layers):
      # Compute new feature maps
      if isDec:
        layer = preact_conv_dec(stack, growth_rate, dropout_p=dropout_p, name=scope + '_' + str(i))
      else:
        layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
      new_features.append(layer)
      # Stack new layer
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features


def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None):
  """
  Transition Down (TD) for FC-DenseNet
  Apply 1x1 BN + ReLU + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as _:
    l = preact_conv(inputs, n_filters, kernel_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l


def TransitionUp(block_to_upsample, n_filters_keep, batch_size, out_res, scope=None):
  """
  Transition Up for FC-DenseNet
  Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
  """
  with tf.name_scope(scope):
    # Upsample
    # print('TransitionUp -->')
    # l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
    # print('before:', block_to_upsample.shape)
    # print('out_res: ', out_res)
    l = deconv2d(block_to_upsample, [batch_size, out_res, out_res, n_filters_keep], k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME', use_spectral_norm=True, name='g_'+scope)
    # print('after:', l.shape)
    # print('TransitionUp <--')

    # Concatenate with skip connection
    # l = tf.concat([l, skip_connection], axis=-1)
    return l


def encoder_dense(inputs, batch_size, feature_size, n_filters_first_conv=48, preset_model='FC-DenseNet56', dropout_p=0.2, scope='g_enc'):
    """
    Builds the FC-DenseNet model

    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
      dropout_p: dropout rate applied after each convolution (0. for not using)
      scope: scope or name

    Returns:
      Fc-DenseNet model
    """
    print('encoder_dense -->')

    if preset_model == 'FC-DenseNet56':
      # FC-DenseNet56: 56 layers, with 4 layers per dense block and a growth rate of 12
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      # growth_rate=16
      growth_rate = 24 # as in DenseNet paper table 1
      n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
      raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % preset_model)

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * n_pool

    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      #####################
      # First Convolution #
      #####################
      # We perform a first convolution.
      # print('inputs before conv: ', inputs.shape)
      stack = slim.conv2d(inputs, n_filters_first_conv, [3, 3], scope='first_conv', activation_fn=None,
                          weights_initializer=tf.random_normal_initializer(stddev=0.02, seed=4285), biases_initializer=tf.constant_initializer(0.01))
      # print('stack after conv: ', stack.shape)

      n_filters = n_filters_first_conv

      #####################
      # Downsampling path #
      #####################

      for i in range(n_pool):
        # Dense Block
        stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, dropout_p, scope='denseblock%d' % (i+1))
        # print('stack after DB: ', stack.shape)
        n_filters += growth_rate * n_layers_per_block[i]

        # Transition Down
        # print('n_filters before TUP:', n_filters)
        stack = TransitionDown(stack, n_filters, dropout_p, scope='transitiondown%d'%(i+1))
        # print('stack after TUP:', stack.shape)

      # 1x1 convolution to reduce channel dimension from 288 to 128
      net = slim.conv2d(stack, 128, [1, 1], activation_fn=None, scope='logits',
                        weights_initializer=tf.random_normal_initializer(stddev=0.02, seed=4285), biases_initializer=tf.constant_initializer(0.01))

      # print('net before reshape:', net.shape)
      net = tf.reshape(net, [batch_size, -1])
      # print('net before reshape:', net.shape)

      assert net.shape[0] == batch_size
      assert net.shape[1] == feature_size

      print('encoder_dense <--')

      return net


def decoder_dense(inputs, batch_size, feature_size, preset_model='FC-DenseNet56', dropout_p=0.2, scope='g_dec', reuse=False):
    """
    Builds the FC-DenseNet model

    Arguments:
      inputs: the input tensor
      preset_model: The model you want to use
      n_classes: number of classes
      n_filters_first_conv: number of filters for the first convolution applied
      n_pool: number of pooling layers = number of transition down = number of transition up
      growth_rate: number of new feature maps created by each layer in a dense block
      n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
      dropout_p: dropout rate applied after each convolution (0. for not using)
      scope: scope or name

    Returns:
      Fc-DenseNet model
    """
    print('decoder_dense -->')

    if preset_model == 'FC-DenseNet56':
      # FC-DenseNet56: 56 layers, with 4 layers per dense block and a growth rate of 12
      n_pool=5
      growth_rate=12
      n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
      n_pool=5
      growth_rate=16
      n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
      n_pool=5
      growth_rate = 24  # as in DenseNet paper table 1
      n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
      n_filters_to_keep = [-1, -1, -1, -1, -1, -1, 656, 464, 304, 192, 112]
      conv2d_res = [-1, -1, -1, -1, -1, -1, 4, 8, 16, 32, 64]
    else:
      raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % preset_model)

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * n_pool
    else:
        raise ValueError

    with tf.variable_scope(scope, preset_model, [inputs]) as sc:

      print('inputs before reshape:', inputs.shape)
      # inputs = tf.reshape(inputs,[batch_size, 1, 1, NUM_TILES_L2_MIX * feature_size])
      # behind the feature rep there are 4 distinct features
      # TODO at the moment there is 1 image only, thus 1x1 feature_size (not 2x2)
      split = feature_size / 4
      assert split.is_integer()
      inputs = tf.reshape(inputs, [batch_size, 2, 2, int(split)])
      print('inputs after reshape:', inputs.shape)

      block_to_upsample = inputs

      #######################
      #   Upsampling path   #
      #######################

      for i in range(n_pool):
        # Self-attention layer (from SAGAN):
        # if i == 4: # before last DB add self-attention layer
        #     #################################
        #     assert n_pool == 5 # designed for case of 5 DBs
        #     print('add self-attention layer SA.. [n_filters_keep: %d]' % block_to_upsample.get_shape()[-1])
        #     block_to_upsample = attention(block_to_upsample, block_to_upsample.get_shape()[-1], sn=True, scope="d_attention", reuse=reuse)
        #     print('block_to_upsample after SA: ', block_to_upsample.shape)
        #     #################################

        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = n_filters_to_keep[n_pool + i + 1]
        out_res = conv2d_res[n_pool + i + 1]
        # print('n_filters_keep TUP:', n_filters_keep)
        stack = TransitionUp(block_to_upsample, n_filters_keep, batch_size, out_res, scope='transitionup%d' % (n_pool + i + 1))
        # print('stack after TUP: ', stack.shape)

        # Dense Block
        # We will only upsample the new feature maps
        n_layers_next = n_layers_per_block[n_pool + i + 1]

        # print('n_layers_next DB:', n_layers_next)
        stack, block_to_upsample = DenseBlock(stack, n_layers_next, growth_rate, dropout_p, isDec=True, scope='denseblock%d' % (n_pool + i + 2))
        # print('stack after DB: ', stack.shape)
        # print('block_to_upsample after DB: ', block_to_upsample.shape)

      num_colors = 3
      # 1x1 convolution to bring down 3rd dimension from 96 to 3
      # net = slim.conv2d(stack, num_colors, [1, 1], activation_fn=None, scope='logits, biases_initializer=tf.constant_initializer(0.01)')
      net = conv2d(stack, num_colors, k_h=1, k_w=1, d_h=1, d_w=1, use_spectral_norm=True, name='logits')

      assert net.shape[0] == batch_size
      assert net.shape[1] == 64
      assert net.shape[2] == 64
      assert net.shape[3] == 3

      print('decoder_dense <--')


      return tf.nn.tanh(net)
