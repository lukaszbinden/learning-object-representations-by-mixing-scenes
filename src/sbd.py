import tensorflow as tf




def decoder_sbd(inputs, image_size, batch_size, feature_size, scope='g_dec_sbd', reuse=False):

    with tf.variable_scope(scope, [inputs]) as _:
        print('decoder_sbd -->')
        print('inputs: %s' % inputs.shape)

        assert inputs.shape[0] == batch_size
        assert inputs.shape[1] == feature_size

        d = w = image_size

        z_b = tf.tile(inputs, [1, d * w])
        print("z_b.shape:", z_b.shape)
        matrix = tf.reshape(z_b, [batch_size, d, w, feature_size])
        print("matrix.shape: ", matrix.shape)

        x = tf.linspace(tf.constant(-1, tf.float32), tf.constant(1, tf.float32), w)
        y = tf.linspace(tf.constant(-1, tf.float32), tf.constant(1, tf.float32), w)

        xb, yb = tf.meshgrid(x, y)
        print("xb.shape: ", xb.shape)

        xb = tf.expand_dims(xb, 2)
        print("xb.shape: ", xb.shape)
        xb = tf.expand_dims(xb, 0)
        print("xb.shape: ", xb.shape)

        ones = tf.ones((batch_size, 64, 64, 1))
        print("ones.shape: ", ones.shape)
        xb = ones * xb
        print("xb.shape: ", xb.shape)

        assert 1 == 0

        xb = tf.concat(axis=0, values=[xb, xb])
        print("xb2.shape: ", xb.shape)
        yb = tf.expand_dims(tf.expand_dims(yb, 2), 0)
        yb = tf.concat(axis=0, values=[yb, yb])

        z_sb = tf.concat(axis=3, values=[matrix, xb, yb])
        print("z_sb.shape:", z_sb.shape)

        assert z_sb.shape[0] == batch_size
        assert z_sb.shape[1] == image_size
        assert z_sb.shape[2] == image_size
        assert z_sb.shape[3] == feature_size + 2

        assert 1 == 0


        net = None

        assert net.shape[0] == batch_size
        assert net.shape[1] == 64
        assert net.shape[2] == 64
        assert net.shape[3] == 3

        print('decoder_sbd <--')

        return tf.nn.tanh(net)
