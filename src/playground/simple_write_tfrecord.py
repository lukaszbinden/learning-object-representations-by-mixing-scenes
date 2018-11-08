from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from random import shuffle
import os, gc, sys
import tensorflow as tf


def create_for_coco_dataset(imgs_dir, annotations_filepath, output_filepath, max_images, shuffle_img=True):
    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds() # totally 82783 images

    if shuffle_img:
        shuffle(img_ids)

    coder = ImageCoder()
    nb_imgs = len(img_ids)
    sess = create_session()

    name = 'val'
    shard = 1
    output_filename = '%s-%.3d-%.1d.tfrecords' % (name, shard, nb_imgs)
    output_file = os.path.join(output_filepath, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    t1_10nn1 = [('000000000927_1.jpg', 0.03125), ('000000568135_2.jpg', 19095.953), ('000000187857_1.jpg', 23359.39),
               ('000000521998_2.jpg', 23557.688), ('000000140816_1.jpg', 24226.852), ('000000015109_1.jpg', 25191.469),
               ('000000525567_1.jpg', 25484.93), ('000000377422_1.jpg', 25654.125), ('000000269815_2.jpg', 26794.836),
               ('000000345617_2.jpg', 26872.812)]

    t1_10nn2 = [('000000000927_1.jpg', 0.031255), ('000000568135_2.jpg', 19095.953), ('000000187857_1.jpg', 23359.39),
               ('000000521998_2.jpg', 23557.688), ('000000140816_1.jpg', 24226.852), ('000000015109_1.jpg', 25191.469),
               ('000000525567_1.jpg', 25484.93), ('000000377422_1.jpg', 25654.125), ('000000269815_2.jpg', 26794.836),
               ('000000345617_2.jpg', 268724.812)]

    img_cnt = 0
    for index, img_id in enumerate(img_ids):
        print('image: ' + str(index))
        if index % 100 == 0:
            print("Reading images: %d / %d "%(index, nb_imgs))
        sys.stdout.flush()

        img_info = process_single_image(img_id, coder, sess, coco, imgs_dir)
        if img_info is None:
            continue

        t1_10nn = t1_10nn1 if img_cnt % 2 == 0 else t1_10nn2
        example = dict_to_coco_example(img_info, t1_10nn)
        writer.write(example.SerializeToString())
        del img_info['pixel_data']
        del img_info
        del example

        img_cnt += 1
        gc.collect()
        if img_cnt >= max_images:
            break

    writer.close()


def create_session():
    sess = tf.Session()
    return sess

def process_single_image(img_id, coder, sess, coco, imgs_dir):
    img_info = {}

    img_detail = coco.loadImgs(img_id)[0]
    pic_height = img_detail['height']
    pic_width = img_detail['width']

    img_path = os.path.join(imgs_dir, img_detail['file_name'])
    del img_detail

    try:
        with tf.gfile.FastGFile(img_path, 'rb') as f:
            image_data = f.read()
            image = coder.decode_jpeg(image_data, sess)
            del image_data
            resized = sess.run(tf.image.resize_images(image, [128, 128]))
            image_data = coder.encode_jpeg(resized, sess)
            del image
    except Exception as e:
        print(e)
        return None

    img_info['pixel_data'] = image_data
    img_info['height'] = pic_height
    img_info['width'] = pic_width

    return img_info


def dict_to_coco_example(img_data, t1_10nn):

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(img_data['height']),
        'image/width': _int64_feature(img_data['width']),
        'image/knn/t1': _bytes_feature(tf.compat.as_bytes(str(t1_10nn))),
        'image/encoded':_bytes_feature(tf.compat.as_bytes(img_data['pixel_data'])),
        'image/format': _bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class ImageCoder(object):

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
    data_dir = '/data/cvg/lukas/datasets/coco/2017_training'
    imgs_dir = os.path.join(data_dir, 'images')
    annotations_filepath = os.path.join(data_dir,'annotations','instances_train2017.json')
    output_filepath = '/data/cvg/lukas/datasets/coco/2017_training/test_tfrecord'
    max_images = 10

    create_for_coco_dataset(imgs_dir, annotations_filepath, output_filepath, max_images, shuffle_img=True)


if __name__ == "__main__":
    tf.app.run()
