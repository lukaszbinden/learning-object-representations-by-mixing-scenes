import tensorflow as tf
import imageio
from skimage.io import imsave

filenames = tf.constant(['..\images\lake.png'])
labels = tf.constant([0])
patch_size = [100, 100]

def parse_fn(file, label):
    image = tf.image.decode_png(tf.read_file(file), channels=3)

    #print(type(image))
    # <class 'tensorflow.python.framework.ops.Tensor'>
    #print(image.get_shape())
    # (?, ?, 3)

    shape= tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    image_channels = shape[2]
    print(image_height)
    print(image_width)
    print(image_channels)
    #num_pieces = tf.div(image_width, image_height)
    num_pieces = 2

    def tf_while_condition(index, outputs):
        # We loop over the number of pieces:
        return tf.less(index, num_pieces)

    def tf_while_body(index, outputs):
        # We get the image patch for the given index:
        sub_image = tf.image.crop_to_bounding_box(image, 0, 10 + (index * 100), 100, 100)
        print('sub_image -->')
        print(type(sub_image))
        print(sub_image.get_shape())
        print('sub_image <--')

        sub_image = tf.expand_dims(sub_image, 0)
        outputs = tf.concat([outputs[:index], sub_image, outputs[index + 1:]], axis=0)
        print(outputs.get_shape())
        outputs.set_shape([2, 100, 100, 3])
        print(outputs.get_shape())
        index = tf.add(index, 1)
        return index, outputs

    # We build our patches tensor which will be filled with the loop:
    patches = tf.zeros((num_pieces, *patch_size, shape[2]), dtype=tf.uint8)
    _, patches = tf.while_loop(tf_while_condition, tf_while_body, [0, patches])

    # We tile the label to have one per patch:
    patches_labels = tf.tile(tf.expand_dims(label, 0), [num_pieces])

    return tf.data.Dataset.from_tensor_slices((patches, patches_labels))


dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.flat_map(parse_fn)

sess = tf.InteractiveSession()
it = dataset.make_one_shot_iterator()
op = it.get_next()
i = 0
while True:
    if i == 2:
        break
    i += 1
    print('while true')
    res = sess.run(op)
    print(type(res))
    # print(res)
    print(type(res[0]))
    print(type(res[1]))
    print(res[1])
    # scipy.misc.imsave('outfile.png', res[0])
    imageio.imwrite(('..\\images\\tile_of_lake_%s.png' % i), res[0])
    print('the end')
