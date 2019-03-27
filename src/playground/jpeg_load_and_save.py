import tensorflow as tf
from utils_common import resize_img
import scipy
from utils_dcgan import save_images, inverse_transform


with tf.Session() as sess:
    tf.set_random_seed(4285)

    file = tf.read_file("../tools/orig_full_000000177006_2.jpg")
    file = tf.image.decode_jpeg(file)
    file = tf.expand_dims(file, 0)
    file = resize_img(file, 64, 1)
    print("file:", file)

    orig = tf.read_file("../tools/orig_000000177006.jpg")
    orig = tf.decode_raw(features['image'], tf.uint8)
    # image.set_shape([img_size * img_size * img_channels])
    orig = tf.image.decode_jpeg(orig)
    orig = tf.image.flip_left_right(orig)
    size = tf.minimum(427, 640)
    crop_shape = tf.parallel_stack([size, size, 3])
    orig = tf.random_crop(orig, crop_shape, seed=4285)
    orig = tf.image.resize_images(orig, [64, 64])
    orig = tf.reshape(orig, (64, 64, 3))
    orig = tf.expand_dims(orig, 0)
    orig = tf.cast(orig, tf.float32) * (2. / 255) - 1
    print("orig:", orig)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    file, orig = sess.run([file, orig])


    scipy.misc.imsave("out/jpeg_load_and_save_1.jpg", file[0])
    file2 = inverse_transform(file)
    scipy.misc.imsave("out/jpeg_load_and_save_2.jpg", file2[0])

    save_images(file, [1,1], "out/jpeg_load_and_save_3.jpg")

    save_images(file, [1,1], "out/jpeg_load_and_save_4.jpg", invert=False)

    scipy.misc.imsave("out/jpeg_load_and_save_5_orig.jpg", orig[0])