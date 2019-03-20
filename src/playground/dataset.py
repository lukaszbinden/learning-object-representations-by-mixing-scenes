import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # str(params.gpu)
import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
from ops_alex import *
import matplotlib.pyplot as plt


EPOCHS = 200
BATCH_SIZE = 256
DATA_DIR = 'D:\\learning-object-representations-by-mixing-scenes\\src\\datasets\\stl-10\\stl10_binary'

with open(DATA_DIR + '\\train_X.bin') as f:
    raw = np.fromfile(f, dtype=np.uint8, count=-1)
    raw = np.reshape(raw, (-1, 3, 96, 96))
    raw = np.transpose(raw, (0, 3, 2, 1))
    X_train_raw = raw

with open(DATA_DIR + '\\train_y.bin') as f:
    raw = np.fromfile(f, dtype=np.uint8, count=-1)
    y_train = raw - 1  # class labels are originally in 1-10 format. Convert them to 0-9 format


# with open(DATA_DIR + '\\test_X.bin') as f:
#     raw = np.fromfile(f, dtype=np.uint8, count=-1)
#     raw = np.reshape(raw, (-1, 3, 96, 96))
#     raw = np.transpose(raw, (0, 3, 2, 1))
#     X_test_raw = raw
#
# with open(DATA_DIR + '\\test_y.bin') as f:
#     raw = np.fromfile(f, dtype=np.uint8, count=-1)
#     y_test = raw - 1



# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(file, label):
  # print(file.shape)
  # print(label.shape)
  # image_string = tf.read_file(filename)
  #image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(file, [64, 64])
  image_resized = tf.cast(image_resized, tf.float32) * (2. / 255) - 1
  return image_resized, label

dataset = tf.data.Dataset.from_tensor_slices((X_train_raw,y_train)).repeat().shuffle(100).batch(BATCH_SIZE)
dataset = dataset.map(_parse_function)

iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()
print("x: ", x.shape)
print("y: ", y.shape)

y = tf.one_hot(y, 10)
print('y one-hot:', y.shape)




# prediction = tf.layers.dense(inputs=x, units=10)
x = tf.reshape(x, [BATCH_SIZE, 64 * 64 * 3])
print("x': ", x.shape)
# prediction = linear(x, 10)
prediction = tf.layers.dense(inputs=x, units=10)
print('prediction: ', prediction.shape)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)
train_op = tf.train.AdamOptimizer().minimize(loss)

# with tf.Session() as sess:
#     img, label = sess.run([x,y])
#     print(img.shape)
#     print(label)

losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))
        losses.append(loss_value)

plt.plot(losses)
plt.show()