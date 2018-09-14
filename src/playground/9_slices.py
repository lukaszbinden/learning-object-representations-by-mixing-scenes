import tensorflow as tf
import imageio
from skimage.io import imsave

files = ['..\\images\\lake.png', '..\\images\\field.png']
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

overlap = 10
side_length = 300

tiles_t1 = tf.image.crop_to_bounding_box(outputs, 0, 30, 150, 150)
tiles_t2 = tf.image.crop_to_bounding_box(outputs, 50, 30, 150, 150)
# tiles_t3 = None
# tiles_t4 = None
# tiles_t5 = None
# tiles_t6 = None
# tiles_t7 = None
# tiles_t8 = None
# tiles_t9 = None




#print(type(image))
# <class 'tensorflow.python.framework.ops.Tensor'>
#print(image.get_shape())
# (?, ?, 3)

# shape= tf.shape(img_1)
# image_height = shape[0]
# image_width = shape[1]
# image_channels = shape[2]
# print(image_height)
# print(image_width)
# print(image_channels)
# #num_pieces = tf.div(image_width, image_height)
# num_pieces = 2
#
# # We get the image patch for the given index:
# sub_image = tf.image.crop_to_bounding_box(img_1, 0, 10 + (index * 100), 100, 100)
# print('sub_image -->')
# print(type(sub_image))
# print(sub_image.get_shape())
# print('sub_image <--')
#
# sub_image = tf.expand_dims(sub_image, 0)
# outputs = tf.concat([outputs[:index], sub_image, outputs[index + 1:]], axis=0)
# print(outputs.get_shape())
# outputs.set_shape([2, 100, 100, 3])
# print(outputs.get_shape())


    # We build our patches tensor which will be filled with the loop:
    # patches = tf.zeros((num_pieces, *patch_size, shape[2]), dtype=tf.uint8)
    #_, patches = tf.while_loop(tf_while_condition, tf_while_body, [0, patches])

    # We tile the label to have one per patch:
    # patches_labels = tf.tile(tf.expand_dims(label, 0), [num_pieces])

    # return tf.data.Dataset.from_tensor_slices((patches, patches_labels))


#dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#dataset = dataset.flat_map(parse_fn)

sess = tf.InteractiveSession()
##it = dataset.make_one_shot_iterator()
#op = it.get_next()

print('sess.run..')
print('tiles_t1 shape: ' + str(tiles_t1.get_shape()))
tiles_t1, tiles_t2 = sess.run([tiles_t1,tiles_t2])
print(type(tiles_t1))
print(tiles_t1.shape)
for idx, img_1 in enumerate(tiles_t1):
    imageio.imwrite(('..\\images\\slices\\slices_t1_%s.png' % idx), img_1)

for idx, img_1 in enumerate(tiles_t2):
    imageio.imwrite(('..\\images\\slices\\slices_t2_%s.png' % idx), img_1)

print('...done.')
