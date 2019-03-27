# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # str(params.gpu)
import sys
sys.path.append('..')
import numpy as np
from ops_alex import *
import matplotlib.pyplot as plt


EPOCHS = 300
BATCH_SIZE = 256
DATA_DIR = 'D:\\learning-object-representations-by-mixing-scenes\\src\\datasets\\stl-10\\stl10_binary'

with open(DATA_DIR + '\\train_X.bin') as f:
    raw = np.fromfile(f, dtype=np.uint8, count=-1)
    raw = np.reshape(raw, (-1, 3, 96, 96))
    raw = np.transpose(raw, (0, 3, 2, 1))
    X_train_raw = raw
    print("X_train_raw.shape: ", X_train_raw.shape)

with open(DATA_DIR + '\\train_y.bin') as f:
    raw = np.fromfile(f, dtype=np.uint8, count=-1)
    y_train = raw - 1  # class labels are originally in 1-10 format. Convert them to 0-9 format
    print("y_train.shape: ", y_train.shape)

with open(DATA_DIR + '\\test_X.bin') as f:
    raw = np.fromfile(f, dtype=np.uint8, count=-1)
    raw = np.reshape(raw, (-1, 3, 96, 96))
    raw = np.transpose(raw, (0, 3, 2, 1))
    X_test_raw = raw

with open(DATA_DIR + '\\test_y.bin') as f:
    raw = np.fromfile(f, dtype=np.uint8, count=-1)
    y_test = raw - 1



# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(file, label):
  image_resized = tf.image.resize_images(file, [64, 64])
  image_resized = tf.cast(image_resized, tf.float32) * (2. / 255) - 1
  return image_resized, label

x_plh = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
y_plh = tf.placeholder(tf.int32, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices((x_plh, y_plh)).repeat().shuffle(100).batch(BATCH_SIZE)
dataset = dataset.map(_parse_function)

iter = dataset.make_initializable_iterator()
x, y = iter.get_next()
print("x: ", x.shape)
print("y: ", y.shape)

y = tf.one_hot(y, 10, dtype=tf.int32)
print('y one-hot:', y.shape)
y = tf.reshape(y, [BATCH_SIZE, 10])
print("y one-hot reshape: ", y.shape)

x = tf.reshape(x, [BATCH_SIZE, 64 * 64 * 3])
print("x': ", x.shape)
prediction = tf.layers.dense(inputs=x, units=10)
print('prediction: ', prediction.shape)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)

_, acc_update_op = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(prediction, axis=1, output_type=tf.int32))
train_op = tf.train.AdamOptimizer().minimize(loss)


##############################################################################################
# TRAINING
##############################################################################################
train_loss_results = []
train_accuracy_results = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(iter.initializer, feed_dict={ x_plh: X_train_raw, y_plh: y_train})
    for i in range(EPOCHS):
        _, loss_value, acc_value = sess.run([train_op, loss, acc_update_op])
        print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss_value, acc_value))
        train_loss_results.append(loss_value)
        train_accuracy_results.append(acc_value)


    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results);
    # plt.show()


##############################################################################################
# TEST
##############################################################################################

# the model only evaluates a single epoch of the test data
    test_loss_results = []
    test_accuracy_results = []
    sess.run(iter.initializer, feed_dict={ x_plh: X_test_raw, y_plh: y_test})
    loss_value, acc_value = sess.run([loss, acc_update_op])
    test_loss_results.append(loss_value)
    test_accuracy_results.append(acc_value)
    print("Test loss: ", loss_value)
    print("Test accuracy:", acc_value)
