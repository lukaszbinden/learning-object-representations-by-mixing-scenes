import tensorflow as tf
import imageio
from skimage.io import imsave

files = ['..\\images\\living_room.jpg', '..\\images\\cat_and_dog.jpg']
patch_size = [300, 300]
num_imgs = 2

img_1 = tf.image.decode_png(tf.read_file(files[0]), channels=3)
shape= tf.shape(img_1)
image_height = shape[0]
image_width = shape[1]
image_channels = shape[2]

img_2 = tf.image.decode_png(tf.read_file(files[1]), channels=3)
outputs = tf.zeros((num_imgs, *patch_size, shape[2]), dtype=tf.uint8)

img_1 = tf.expand_dims(img_1, 0)
img_2 = tf.expand_dims(img_2, 0)
outputs = tf.concat([outputs[:0], img_1, outputs[0 + 1:]], axis=0)
outputs = tf.concat([outputs[:1], img_2, outputs[1 + 1:]], axis=0)
print('outputs: ' + str(outputs.get_shape()))

overlap = 30 # hyperparameter
# assert overlap, 'hyperparameter \'overlap\' is not an integer'
image_size = 300
slice_size = (image_size + 2 * overlap) / 3
slice_size_overlap = slice_size - overlap
assert slice_size.is_integer(), 'hyperparameter \'overlap\' invalid'
slice_size = int(slice_size)
slice_size_overlap = int(slice_size_overlap)

tile_r1c1 = tf.image.crop_to_bounding_box(outputs, 0, 0, slice_size, slice_size)
tile_r1c2 = tf.image.crop_to_bounding_box(outputs, 0, slice_size_overlap, slice_size, slice_size)
tile_r1c3 = tf.image.crop_to_bounding_box(outputs, 0, image_size - slice_size, slice_size, slice_size)
tile_r2c1 = tf.image.crop_to_bounding_box(outputs, slice_size_overlap, 0, slice_size, slice_size)
tile_r2c2 = tf.image.crop_to_bounding_box(outputs, slice_size_overlap, slice_size_overlap, slice_size, slice_size)
tile_r2c3 = tf.image.crop_to_bounding_box(outputs, slice_size_overlap, image_size - slice_size, slice_size, slice_size)
tile_r3c1 = tf.image.crop_to_bounding_box(outputs, image_size - slice_size, 0, slice_size, slice_size)
tile_r3c2 = tf.image.crop_to_bounding_box(outputs, image_size - slice_size, slice_size_overlap, slice_size, slice_size)
tile_r3c3 = tf.image.crop_to_bounding_box(outputs, image_size - slice_size, image_size - slice_size, slice_size, slice_size)


sess = tf.InteractiveSession()

print('sess.run..')
print('tile_r1c1 shape: ' + str(tile_r1c1.get_shape()))
tile_r1c1,tile_r1c2,tile_r1c3,tile_r2c1,tile_r2c2,tile_r2c3,tile_r3c1,tile_r3c2,tile_r3c3 = sess.run([tile_r1c1,tile_r1c2,tile_r1c3,tile_r2c1,tile_r2c2,tile_r2c3,tile_r3c1,tile_r3c2,tile_r3c3])
print(type(tile_r1c1))
print(tile_r1c1.shape)
for idx, img_1 in enumerate(tile_r1c1):
    imageio.imwrite(('..\\images\\slices\\tile_r1c1_%s.png' % idx), img_1)
for idx, img_1 in enumerate(tile_r1c2):
    imageio.imwrite(('..\\images\\slices\\tile_r1c2_%s.png' % idx), img_1)
for idx, img_1 in enumerate(tile_r1c3):
    imageio.imwrite(('..\\images\\slices\\tile_r1c3_%s.png' % idx), img_1)
for idx, img_1 in enumerate(tile_r2c1):
    imageio.imwrite(('..\\images\\slices\\tile_r2c1_%s.png' % idx), img_1)
for idx, img_1 in enumerate(tile_r2c2):
    imageio.imwrite(('..\\images\\slices\\tile_r2c2_%s.png' % idx), img_1)
for idx, img_1 in enumerate(tile_r2c3):
    imageio.imwrite(('..\\images\\slices\\tile_r2c3_%s.png' % idx), img_1)
for idx, img_1 in enumerate(tile_r3c1):
    imageio.imwrite(('..\\images\\slices\\tile_r3c1_%s.png' % idx), img_1)
for idx, img_1 in enumerate(tile_r3c2):
    imageio.imwrite(('..\\images\\slices\\tile_r3c2_%s.png' % idx), img_1)
for idx, img_1 in enumerate(tile_r3c3):
    imageio.imwrite(('..\\images\\slices\\tile_r3c3_%s.png' % idx), img_1)

print('...done.')
