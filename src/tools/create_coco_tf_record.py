r"""Convert raw Microsoft COCO dataset to TFRecord for object_detection.
Attention Please!!!

1)For easy use of this script, Your coco dataset directory struture should like this :
    +Your coco dataset root
        +train2017
        +val2017
        +annotations
            -instances_train2017.json
            -instances_val2017.json
2)To use this script, you should download python coco tools from "http://mscoco.org/dataset/#download" and make it.
After make, copy the pycocotools directory to the directory of this "create_coco_tf_record.py"
or add the pycocotools path to  PYTHONPATH of ~/.bashrc file.

Example usage:
    python create_coco_tf_record.py --data_dir=/path/to/your/coco/root/directory \
        --set=train \
        --output_path=/where/you/want/to/save/pascal.record
        --shuffle_imgs=True

Source: https://github.com/MetaPeak/tensorflow_object_detection_create_coco_tfrecord
Modified by Lukas Zbinden (image resizing).
Requires file 'build_image_data.py' (class ImageDecoder)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from random import shuffle
import os, gc, sys
import tensorflow as tf

import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/data/cvg/lukas/datasets/coco', 'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('set', 'val', 'Convert training set or validation set')
flags.DEFINE_string('output_filepath', '/data/cvg/lukas/datasets/coco/2017_val/tfrecords', 'Path to output TFRecord')
flags.DEFINE_bool('shuffle_imgs',True,'whether to shuffle images of coco')
flags.DEFINE_integer('images_per_file', 2048, 'Number of images per file. [2048]')
flags.DEFINE_integer('image_size', 300, 'Excpected width and length of all images, [227]')
FLAGS = flags.FLAGS


def create_for_coco_dataset(imgs_dir, annotations_filepath, shuffle_img = True):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
        shuffle_img: wheter to shuffle images order
    Return:
        coco_data: list of dictionary format information of each image
    """
    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds() # totally 82783 images
    cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.

    if shuffle_img:
        shuffle(img_ids)

    coder = ImageCoder()
    nb_imgs = len(img_ids)
    sess = tf.Session()

    name = FLAGS.set
    shard = 1
    output_filename = '%s-%.3d-%.1d-%.1d.tfrecords' % (name, shard, FLAGS.images_per_file, nb_imgs)
    output_file = os.path.join(FLAGS.output_filepath, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    img_cnt = 0
    for index, img_id in enumerate(img_ids):
        print('image: ' + str(index))

        if index % 100 == 0:
            print("Reading images: %d / %d "%(index, nb_imgs))
        sys.stdout.flush()
        img_info = process_single_image(img_id, coder, sess, coco, cat_ids, imgs_dir)
        if img_info is None:
            continue

        example = dict_to_coco_example(img_info)
        writer.write(example.SerializeToString())
        del img_info

        img_cnt += 1
        if img_cnt % FLAGS.images_per_file == 0:
            print('close current writer and create new one... #shard: ' + str(shard))
            sys.stdout.flush()
            writer.close()
            del writer
            gc.collect()
            shard += 1
            output_filename = '%s-%.3d-%.1d-%.1d.tfrecords' % (name, shard, FLAGS.images_per_file, nb_imgs)
            output_file = os.path.join(FLAGS.output_filepath, output_filename)
            writer = tf.python_io.TFRecordWriter(output_file)

    writer.close()

def process_single_image(img_id, coder, sess, coco, cat_ids, imgs_dir):
    img_info = {}
    bboxes = []
    labels = []

    img_detail = coco.loadImgs(img_id)[0]
    pic_height = img_detail['height']
    pic_width = img_detail['width']

    ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        bboxes_data = ann['bbox']
        bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),\
                              bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                     # the format of coco bounding boxs is [Xmin, Ymin, width, height]
        bboxes.append(bboxes_data)
        labels.append(ann['category_id'])

    img_path = os.path.join(imgs_dir, img_detail['file_name'])

    try:
        with tf.gfile.FastGFile(img_path, 'rb') as f:
            image_data = f.read()
            image = coder.decode_jpeg(image_data, sess)
            resized = sess.run(
                tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size]))

            image_data = coder.encode_jpeg(resized, sess)
            assert len(image.shape) == 3
            assert pic_height == image.shape[0]
            assert pic_width == image.shape[1]
            assert image.shape[2] == 3
            del image
    except Exception as e:
        print(e)
        return None

    img_info['pixel_data'] = image_data
    img_info['height'] = pic_height
    img_info['width'] = pic_width
    img_info['bboxes'] = bboxes
    img_info['labels'] = labels

    return img_info


def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(img_data['height']),
        'image/width': dataset_util.int64_feature(img_data['width']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(img_data['labels']),
        'image/encoded': dataset_util.bytes_feature(tf.compat.as_bytes(img_data['pixel_data'])),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example

#
# Modified class ImageCoder. Original version from:
# https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
#
class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data)

  def png_to_jpeg(self, image_data, sess):
    return sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data, sess):
    image = sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def encode_jpeg(self, image, sess):
    image_data = sess.run(self._encode_jpeg,
                           feed_dict={self._encode_jpeg_data: image})
    return image_data


def main(_):
    if FLAGS.set == "train":
        data_dir = os.path.join(FLAGS.data_dir, '2017_training')
        imgs_dir = os.path.join(data_dir, 'images')
        annotations_filepath = os.path.join(data_dir,'annotations','instances_train2017.json')
        print("Convert coco train file to tf record")
    elif FLAGS.set == "val":
        data_dir = os.path.join(FLAGS.data_dir, '2017_val')
        imgs_dir = os.path.join(data_dir, 'images')
        annotations_filepath = os.path.join(data_dir,'annotations','instances_val2017.json')
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")

    create_for_coco_dataset(imgs_dir, annotations_filepath, shuffle_img=FLAGS.shuffle_imgs)


if __name__ == "__main__":
    tf.app.run()
