import tensorflow as tf

filenames = tf.constant(['..\images\lake.png', '..\images\field.png'])
labels = tf.constant([0, 1])
patch_size = [100, 100]

def parse_fn(file, label):
    image = tf.image.decode_png(tf.read_file(file), channels=3)

    print(type(image))
    # <class 'tensorflow.python.framework.ops.Tensor'>
    print(image.get_shape())
    # (?, ?, 3)

    shape= tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    print(image_height)
    print(image_width)
    #num_pieces = tf.div(image_width, image_height)
    num_pieces = 9

    def tf_while_condition(index, outputs):
        # We loop over the number of pieces:
        return tf.less(index, num_pieces)

    def tf_while_body(index, outputs):
        # We get the image patch for the given index:
        offset_width = index * image_height
        sub_image = tf.image.crop_to_bounding_box(image, 0, offset_width, image_height, image_height)
        sub_image = tf.image.resize_images(sub_image, patch_size)
        sub_image = tf.expand_dims(sub_image, 0)
        # We add it to the output patches (may be a cleaner way to do so):
        outputs = tf.concat([outputs[:index], sub_image, outputs[index + 1:]], axis=0)
        # We increment our index and return the values:
        index = tf.add(index, 1)
        return index, outputs

    # We build our patches tensor which will be filled with the loop:
    patches = tf.zeros((num_pieces, *patch_size, shape[2]), dtype=tf.float32)
    _, patches = tf.while_loop(tf_while_condition, tf_while_body, [0, patches])

    # We tile the label to have one per patch:
    patches_labels = tf.tile(tf.expand_dims(label, 0), [num_pieces])

    return tf.data.Dataset.from_tensor_slices((patches, patches_labels))


dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.flat_map(parse_fn)

sess = tf.InteractiveSession()
it = dataset.make_one_shot_iterator()
op = it.get_next()
while True:
    res = sess.run(op)
    print(res)
