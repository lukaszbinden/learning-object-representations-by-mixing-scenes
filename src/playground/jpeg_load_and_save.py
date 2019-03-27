import tensorflow as tf
from utils_common import resize_img
import scipy
from utils_dcgan import save_images, inverse_transform
import imageio


def prep(orig):
    orig = tf.reshape(orig, (64, 64, 3))
    orig = tf.expand_dims(orig, 0)
    orig = tf.cast(orig, tf.float32) * (2. / 255) - 1
    return orig

def imwrite(path, img):
    imageio.imwrite(path, img)
    # scipy.misc.imsave(path, img)


with tf.Session() as sess:
    tf.set_random_seed(4285)

    OUT_DIR = "out/000000410912/imwrite_vs_scipy"

    # file = tf.read_file("../tools/orig_full_000000177006_2.jpg")
    file = tf.read_file("../tools/orig_full_000000410912_2.jpg")
    file = tf.image.decode_jpeg(file)
    file = tf.expand_dims(file, 0)
    file = resize_img(file, 64, 1)
    print("file:", file)

    # orig = tf.read_file("../tools/orig_000000177006.jpg")
    orig = tf.read_file("../tools/orig_000000410912.jpg")
    orig = tf.image.decode_jpeg(orig)
    orig = tf.image.flip_left_right(orig)
    size = tf.minimum(427, 640)
    crop_shape = tf.parallel_stack([size, size, 3])
    orig = tf.random_crop(orig, crop_shape, seed=4285)
    origimg = orig
    orig = None

    orig = tf.image.resize_images(origimg, [64, 64], method=tf.image.ResizeMethod.BILINEAR)
    orig = prep(orig)
    print("orig:", orig)
    orignn = tf.image.resize_images(origimg, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    orignn = prep(orignn)
    origbi = tf.image.resize_images(origimg, [64, 64], method=tf.image.ResizeMethod.BICUBIC)
    origbi = prep(origbi)
    origar = tf.image.resize_images(origimg, [64, 64], method=tf.image.ResizeMethod.AREA)
    origar = prep(origar)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    file, orig, nn, bi, ar = sess.run([file, orig, orignn, origbi, origar])


    scipy.misc.imsave(OUT_DIR + "/jpeg_load_and_save_1.png", file[0])
    file2 = inverse_transform(file)
    scipy.misc.imsave(OUT_DIR + "/jpeg_load_and_save_2.png", file2[0])

    save_images(file, [1,1], OUT_DIR + "/jpeg_load_and_save_3.png")

    save_images(file, [1,1], OUT_DIR + "/jpeg_load_and_save_4.png", invert=False)

    imwrite(OUT_DIR + "/jpeg_load_and_save_5_bil_im.png", orig[0])
    imwrite(OUT_DIR + "/jpeg_load_and_save_6_nn_im.png", nn[0])
    imwrite(OUT_DIR + "/jpeg_load_and_save_7_bic_im.png", bi[0])
    imwrite(OUT_DIR + "/jpeg_load_and_save_8_ar_im.png", ar[0])

    scipy.misc.imsave(OUT_DIR + "/jpeg_load_and_save_5_bil.png", orig[0])
    scipy.misc.imsave(OUT_DIR + "/jpeg_load_and_save_6_nn.png", nn[0])
    scipy.misc.imsave(OUT_DIR + "/jpeg_load_and_save_7_bic.png", bi[0])
    scipy.misc.imsave(OUT_DIR + "/jpeg_load_and_save_8_ar.png", ar[0])

