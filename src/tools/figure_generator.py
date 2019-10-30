import sys
sys.path.append('..')
import scipy.misc
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
from utils_dcgan import save_images_6cols, save_images_7cols
import random
from datetime import datetime
import tensorflow as tf

img_size = 64
num = 5
# out = np.ones((img_size*(num+1),img_size*(num+1),3))*255
# chunk_num = 8

input = 'C:\\Users\\lz826\\Dropbox\\Master Thesis\\Thesis writing\\images\\images\\data_augmentation\\ex2_190214\\tmp'
output = "result"
output = os.path.join(input, output)
output = os.path.join(output, datetime.now().strftime('%Y%m%d_%H%M%S'))
if not os.path.exists(output):
	os.makedirs(output)

onlyfiles = [join(input, f) for f in listdir(input) if isfile(join(input, f))]
shuffled_index = list(range(len(onlyfiles)))
#random.seed(4285)
random.shuffle(shuffled_index)
onlyfiles = [onlyfiles[i] for i in shuffled_index]
# print(onlyfiles)

grid = [1, num]

# imgs = []
# for m in range(num):
# 	img_path = onlyfiles[m]
# 	img = cv2.imread(img_path)
# 	b, g, r = cv2.split(img) # get b,g,r channels
# 	img = cv2.merge([r,g,b])
# 	if img_size != img.shape[0]:
# 		img = cv2.resize(img, (img_size, img_size))
# 		print('image resized: ', img.shape)
# 	imgs.append(img)

# step 1
filenames = tf.constant(onlyfiles)
labels = tf.constant(np.ones((len(onlyfiles))), dtype=tf.uint8)

# step 2: create a dataset returning slices of `filenames`
print(filenames.shape)
print(labels.shape)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# step 3: parse every image in the dataset using `map`
def parse_function(name, label):
	image_string = tf.read_file(name)
	image_decoded = tf.image.decode_jpeg(image_string, channels=3)
	image_resized = tf.image.resize_images(image_decoded, [64, 64])
	image = tf.cast(image_resized, tf.float32)
	return image, label

dataset = dataset.map(parse_function)
dataset = dataset.batch(5)

# step 4: create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
images, _ = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(len(onlyfiles))
    # for _ in range(len(onlyfiles)):
    igs = sess.run([images])
    imgs = igs[0]

save_path = os.path.join(output, "result.png")
print(imgs[0].shape)
save_images_6cols(imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], None, grid, None, save_path, addSpacing=4)
# save_images_6cols(imgs[0], imgs[1], None, None, None, None, grid, 1, save_path, maxImg=1)
print("saved images to %s" % save_path)




