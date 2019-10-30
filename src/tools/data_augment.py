import tensorflow as tf

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, FLAGS):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=0) # 0 = Use the number of channels in the PNG-encoded image.
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=0)

    self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data)

    self._resize_jpeg_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
    self._resize_jpeg = tf.image.resize_images(self._resize_jpeg_data, [FLAGS.image_size, FLAGS.image_size])

    self._flip_left_right_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
    self._flip_left_right = tf.image.flip_left_right(self._flip_left_right_data)

    self._crop_jpeg_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
    self._crop_jpeg = tf.random_crop(self._crop_jpeg_data, [FLAGS.image_size, FLAGS.image_size, 3], seed=4285)

    self.f = tf.placeholder(dtype=tf.int32, shape=())
    self.h = tf.placeholder(dtype=tf.int32, shape=())
    self.w = tf.placeholder(dtype=tf.int32, shape=())
    height_s = tf.cast(tf.round(tf.divide(tf.multiply(self.h, self.f), 10)), tf.int32)
    width_s = tf.cast(tf.round(tf.divide(tf.multiply(self.w, self.f), 10)), tf.int32)
    crop_shape = tf.parallel_stack([height_s, width_s, 3])
    self._scale_jpeg_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
    self._scale_jpeg = tf.random_crop(self._scale_jpeg_data, crop_shape, seed=4285)

    self.rc_h = tf.placeholder(dtype=tf.int32, shape=())
    self.rc_w = tf.placeholder(dtype=tf.int32, shape=())
    size = tf.minimum(self.rc_h, self.rc_w)
    rc_crop_shape = tf.parallel_stack([size, size, 3])
    self._random_crop_jpeg_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
    self._random_crop_jpeg = tf.random_crop(self._random_crop_jpeg_data, rc_crop_shape, seed=4285)

  def flip_left_right(self, image):
    flipped = self._sess.run(self._flip_left_right,
                           feed_dict={self._flip_left_right_data: image})
    return flipped

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    return image

  def encode_jpeg(self, image):
    image_data = self._sess.run(self._encode_jpeg,
                           feed_dict={self._encode_jpeg_data: image})
    return image_data

  def resize(self, image):
    resized = self._sess.run(self._resize_jpeg,
                           feed_dict={self._resize_jpeg_data: image})
    return resized

  def scale(self, image, height, width, factor):
    image_scaled = self._sess.run(self._scale_jpeg, feed_dict={self._scale_jpeg_data: image,
                                                               self.f: factor, self.h: height, self.w: width})
    return image_scaled

  def random_crop_max(self, image, height, width):
    cropped = self._sess.run(self._random_crop_jpeg,
                           feed_dict={self._random_crop_jpeg_data: image, self.rc_h: height, self.rc_w: width})
    return cropped

  def crop(self, image):
    cropped = self._sess.run(self._crop_jpeg,
                           feed_dict={self._crop_jpeg_data: image})
    return cropped


def augment_image(image_data, image, height, width, coder, apply_data_augment=True):
    result = []
    heights = []
    widths = []
    images = []

    result.append(image_data)
    heights.append(height)
    widths.append(width)
    images.append(image)

    if apply_data_augment:
        # ----------------------------
        # for _ in range(FLAGS.num_crops):
        #  crop = coder.crop(image)
        #  image_data = coder.encode_jpeg(crop)
        #  result.append(image_data)

        # flip ----------------------------
        flipped = coder.flip_left_right(image)
        image_data = coder.encode_jpeg(flipped)
        result.append(image_data)
        flipped_height = flipped.shape[0]
        heights.append(flipped_height)
        flipped_width = flipped.shape[1]
        widths.append(flipped_width)
        images.append(flipped)

        # scale with 0.6 ------------------
        scale_image(coder, result, heights, widths, images, height, image, width, [6])
        scale_image(coder, result, heights, widths, images, flipped_height, flipped, flipped_width, [7])

        # 1x random crop each ------------------
        # random_crop_max(coder, image, result, heights, widths, images, height, width)
        # random_crop_max(coder, flipped, result, heights, widths, images, flipped_height, flipped_width)
        scale_image(coder, result, heights, widths, images, flipped_height, flipped, flipped_width, [9.5])

        assert len(result) == len(heights) and len(heights) == len(widths)
        assert len(result) == 5

    for h in heights:
        assert type(h) == int, str(heights)

    for w in widths:
        assert type(w) == int, str(widths)

    return result, heights, widths, images


def random_crop_max(coder, image, result, heights, widths, images, height, width):
    rc2 = coder.random_crop_max(image, height, width)
    image_data = coder.encode_jpeg(rc2)
    result.append(image_data)
    heights.append(rc2.shape[0])
    widths.append(rc2.shape[1])
    images.append(rc2)


def scale_image(coder, result, heights, widths, images, height, image, width, factors):
  for factor in factors: # add more factors if required
    cropped = coder.scale(image, height, width, factor)
    image_data = coder.encode_jpeg(cropped)
    result.append(image_data)
    heights.append(cropped.shape[0])
    widths.append(cropped.shape[1])
    images.append(cropped)