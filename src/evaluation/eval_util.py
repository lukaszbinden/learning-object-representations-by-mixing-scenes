import tensorflow as tf

def resize_scale_w(imag, ih, iw):
    # w = int(float(256 * iw) / ih)
    r = tf.cast(256 * iw, tf.float32)
    ihf = tf.cast(ih, tf.float32)
    w = tf.cast(tf.div(r, ihf), tf.int32)
    shape = tf.parallel_stack([256, w, 3])
    imag = tf.expand_dims(imag, 0)
    imag = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
                    true_fn=lambda: tf.image.resize_bilinear(imag, shape[:2], align_corners=False),
                    false_fn=lambda: tf.image.resize_bicubic(imag, shape[:2], align_corners=False))
    imag = tf.squeeze(imag)
    return tf.reshape(imag, (256, w, 3), name="resize_scale_w_reshape")


def resize_scale_h(imag, ih, iw):
    # h = int(float(256 * ih) / iw)
    r = tf.cast(256 * ih, tf.float32)
    iwf = tf.cast(iw, tf.float32)
    h = tf.cast(tf.div(r, iwf), tf.int32)
    shape = tf.parallel_stack([h, 256, 3])
    imag = tf.expand_dims(imag, 0)
    imag = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
                    true_fn=lambda: tf.image.resize_bilinear(imag, shape[:2], align_corners=False),
                    false_fn=lambda: tf.image.resize_bicubic(imag, shape[:2], align_corners=False))
    imag = tf.squeeze(imag)
    return tf.reshape(imag, (h, 256, 3), name="resize_scale_h_reshape")
