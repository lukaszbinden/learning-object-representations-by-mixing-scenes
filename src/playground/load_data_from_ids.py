import tensorflow as tf
import os
import numpy as np

# # step 1
# filenames = tf.constant(['im_01.jpg', 'im_02.jpg', 'im_03.jpg', 'im_04.jpg'])
# labels = tf.constant([0, 1, 0, 1])
#
# # step 2: create a dataset returning slices of `filenames`
# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#
# # step 3: parse every image in the dataset using `map`
# def _parse_function(filename, label):
#     image_string = tf.read_file(filename)
#     image_decoded = tf.image.decode_jpeg(image_string, channels=3)
#     image = tf.cast(image_decoded, tf.float32)
#     return image, label
#
# dataset = dataset.map(_parse_function)
# dataset = dataset.batch(2)
#
# # step 4: create iterator and final input tensor
# iterator = dataset.make_one_shot_iterator()
# images, labels = iterator.get_next()


DATA_DIR = 'D:\\learning-object-representations-by-mixing-scenes\\src\\datasets\\stl-10\\stl10_binary'

with open(os.path.join(DATA_DIR, 'fold_indices.txt')) as f:
    raw = np.loadtxt(f, dtype=np.uint32)
    X_train_raw = raw
    print("X_train_raw.shape: ", type(X_train_raw[0]))
    print(X_train_raw[0])