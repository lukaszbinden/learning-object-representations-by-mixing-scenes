'''
Author: LZ, 02.04.19

Usage: python resize_jpeg.py  /home/lz01a008/results/exp73/test_images/test/57/img_mix_gen_10488.png

Also see shortcut in ~/bin/rjp

'''

import sys
# sys.path.append('..')
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from scipy.misc import imsave

IMG_SIZE = 64
OUT_DIR = "png"


def resize_jpeg(argv):
    # print(argv)
    assert len(argv) == 2, "just 2 args"

    input_dir = argv[1]

    out_dir = os.path.join(input_dir, OUT_DIR)
    # output = os.path.join(output, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fnames = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith(".jpg")]
    files = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith(".jpg")]
    num_files = len(files)
    print("found %d files in '%s'." % (num_files, input_dir))

    filenames = tf.constant(files)
    fnames = tf.constant(fnames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, fnames))

    # step 3: parse every image in the dataset using `map`
    def parse_function(name, fname):
        image_string = tf.read_file(name)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
        image_resized = tf.reshape(image_resized, (IMG_SIZE, IMG_SIZE, 3))
        image = tf.cast(image_resized, tf.float32)
        return image, fname

    dataset = dataset.map(parse_function)
    dataset = dataset.batch(num_files)

    iterator = dataset.make_one_shot_iterator()
    images, fname = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        imgs, f_name = sess.run([images, fname])

    for i in range(num_files):
        img = imgs[i]
        fn = f_name[i]
        fn = fn.decode("utf-8").split(".")[0] + ".png"
        fn = os.path.join(out_dir, fn)
        imsave(fn, img)

    print("saved %d images to %s" % (num_files, out_dir))


if __name__ == '__main__':
    resize_jpeg(argv=sys.argv)