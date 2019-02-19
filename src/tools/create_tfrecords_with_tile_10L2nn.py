# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

LZ: modified, 10-11/2018
Original Src: https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
Original Src: https://github.com/MetaPeak/tensorflow_object_detection_create_coco_tfrecord
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import time
import pickle

import numpy as np
import tensorflow as tf
import traceback
from pycocotools.coco import COCO
from data_augment import ImageCoder, augment_image

tf.app.flags.DEFINE_string('train_directory', '/data/cvg/lukas/datasets/coco/2017_training/',
                           'Training data directory')
tf.app.flags.DEFINE_string('train_ann_file', 'instances_train2017.json',
                           'Training data annotation file')
tf.app.flags.DEFINE_string('validation_directory', '/data/cvg/lukas/datasets/coco/2017_val/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('val_ann_file', 'instances_val2017.json',
                           'Validation data annotation file')
tf.app.flags.DEFINE_string('train_output_directory', '/data/cvg/lukas/datasets/coco/2017_training/version/v5/final/',
                           'Train Output data directory')
tf.app.flags.DEFINE_string('val_output_directory', '/data/cvg/lukas/datasets/coco/2017_val/version/v5/final/',
                           'Validation Output data directory')
tf.app.flags.DEFINE_string('basedir_knn_lists', '/data/cvg/lukas/deepcluster/main_coco_out/tile_clustering/2017_training/version/v5/',
                           'Tile to knn dict directory')

tf.app.flags.DEFINE_integer('train_shards', 150,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 6,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 10,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('image_size', 200,
                            'Excpected width and length of all images, [300]')
tf.app.flags.DEFINE_integer('min_num_bbox', 4,
                            'Minimum number of bounding boxes / objects, [5]')
tf.app.flags.DEFINE_boolean('do_flip', True,
                            'If each image should be flipped, [True]')
tf.app.flags.DEFINE_integer('num_crops', 0,
                            'Number of crops per image, [3]')
tf.app.flags.DEFINE_integer('num_images', None,
                            'Number of images to use (incl. flips), None -> all')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list =tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width, t_to_10nn_dict):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  filename = os.path.basename(filename)
  try:
    t1_ids, t1_sub_ids, t1_L2 = get_ids(t_to_10nn_dict['t1'][filename])
    t2_ids, t2_sub_ids, t2_L2 = get_ids(t_to_10nn_dict['t2'][filename])
    t3_ids, t3_sub_ids, t3_L2 = get_ids(t_to_10nn_dict['t3'][filename])
    t4_ids, t4_sub_ids, t4_L2 = get_ids(t_to_10nn_dict['t4'][filename])
  except Exception as e:
    print('some 10nn list not found for filename=%s...' % filename)
    print(e)
    tb = traceback.format_exc()
    print(tb)
    print('thus not adding this file to tfrecorcds...')
    return None

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
      'image/knn/t1': _int64_feature(t1_ids),
      'image/knn/t1s': _int64_feature(t1_sub_ids),
      'image/knn/t1L2': _float_feature(t1_L2),
      'image/knn/t2': _int64_feature(t2_ids),
      'image/knn/t2s': _int64_feature(t2_sub_ids),
      'image/knn/t2L2': _float_feature(t2_L2),
      'image/knn/t3': _int64_feature(t3_ids),
      'image/knn/t3s': _int64_feature(t3_sub_ids),
      'image/knn/t3L2': _float_feature(t3_L2),
      'image/knn/t4': _int64_feature(t4_ids),
      'image/knn/t4s': _int64_feature(t4_sub_ids),
      'image/knn/t4L2': _float_feature(t4_L2),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example


def get_ids(t_10nn):
  ids = []
  sub_ids = []
  L2 = []
  for tupl in t_10nn:
    spl = tupl[0].split('_')
    ids.append(int(spl[0]))
    sub_ids.append(int(spl[1].split('.')[0]))
    L2.append(tupl[1])
  return ids, sub_ids, L2


# class ImageCoder(object):
#   """Helper class that provides TensorFlow image coding utilities."""
#
#   def __init__(self):
#     # Create a single Session to run all image coding calls.
#     self._sess = tf.Session()
#
#     tf.set_random_seed(4285)
#
#     # Initializes function that converts PNG to JPEG data.
#     self._png_data = tf.placeholder(dtype=tf.string)
#     image = tf.image.decode_png(self._png_data, channels=0) # 0 = Use the number of channels in the PNG-encoded image.
#     self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
#
#     # Initializes function that decodes RGB JPEG data.
#     self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
#     self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=0)
#
#     self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
#     self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data)
#
#     self._resize_jpeg_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
#     self._resize_jpeg = tf.image.resize_images(self._resize_jpeg_data, [FLAGS.image_size, FLAGS.image_size])
#
#     self._flip_left_right_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
#     self._flip_left_right = tf.image.flip_left_right(self._flip_left_right_data)
#
#     self._crop_jpeg_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
#     self._crop_jpeg = tf.random_crop(self._crop_jpeg_data, [FLAGS.image_size, FLAGS.image_size, 3], seed=4285)
#
#   def flip_left_right(self, image):
#     flipped = self._sess.run(self._flip_left_right,
#                            feed_dict={self._flip_left_right_data: image})
#     return flipped
#
#   def png_to_jpeg(self, image_data):
#     return self._sess.run(self._png_to_jpeg,
#                           feed_dict={self._png_data: image_data})
#
#   def decode_jpeg(self, image_data):
#     image = self._sess.run(self._decode_jpeg,
#                            feed_dict={self._decode_jpeg_data: image_data})
#     assert len(image.shape) == 3
#     return image
#
#   def encode_jpeg(self, image):
#     image_data = self._sess.run(self._encode_jpeg,
#                            feed_dict={self._encode_jpeg_data: image})
#     return image_data
#
#   def resize(self, image):
#     resized = self._sess.run(self._resize_jpeg,
#                            feed_dict={self._resize_jpeg_data: image})
#     return resized
#
#   def crop(self, image):
#     cropped = self._sess.run(self._crop_jpeg,
#                            feed_dict={self._crop_jpeg_data: image})
#     return cropped


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  # del image_data

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3, 'image %s not RGB but %d' % (filename, image.shape[2])

  # check that image is large enough for our purposes
  if height < FLAGS.image_size or width < FLAGS.image_size:
      del image
      return None, height, width

  assert FLAGS.num_crops > 0 or FLAGS.do_flip

  assert FLAGS.num_crops == 0, 'num_crops not supported at the moment'

  # result = []
  #
  # for _ in range(FLAGS.num_crops):
  #   crop = coder.crop(image)
  #   image_data = coder.encode_jpeg(crop)
  #   result.append(image_data)
  #
  # if FLAGS.do_flip:
  #   flipped = coder.flip_left_right(image)
  #   result.append(image_data)
  #   image_data = coder.encode_jpeg(flipped)
  #   result.append(image_data)
  #
  # return result, height, width

  result, heights, widths, _ = augment_image(image_data, image, height, width, coder)
  return result, heights, widths


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, num_shards, t_to_10nn_dict):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
  if FLAGS.num_crops > 1:
    num_files_in_thread *= FLAGS.num_crops

  counter = 0
  img_error = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d.tfrecords' % (name, (shard + 1), num_shards)
    output_dir = FLAGS.train_output_directory if name is 'train' else FLAGS.val_output_directory
    output_file = os.path.join(output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]

      try:
        image_buffers, heights, widths = _process_image(filename, coder)
        if image_buffers is None:
            #print('image %s too small' % filename)
            continue
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected error while decoding %s.' % filename)
        continue

      img_id = 1
      for ind in range(len(image_buffers)):
        image_buffer = image_buffers[ind]
        height = heights[ind]
        width = widths[ind]
        fn = filename.split('.jpg')[0] + '_' + str(img_id) + '.jpg'
        img_id += 1
        example = _convert_to_example(fn, image_buffer, height, width, t_to_10nn_dict)
        if example:
          writer.write(example.SerializeToString())
          shard_counter += 1
          counter += 1
        else:
          img_error += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s  [img_errors: %d]' %
          (datetime.now(), thread_index, shard_counter, output_file, img_error))
    sys.stdout.flush()
  print('%s [thread %d]: Wrote %d images from %d files [img_errors: %d].' %
        (datetime.now(), thread_index, counter, num_files_in_thread, img_error))
  sys.stdout.flush()


def _process_image_files(name, filenames, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    num_shards: integer number of shards for this data set.
  """

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder(FLAGS)

  print('unpickle clustering objects...')
  basedir_knn_lists = FLAGS.basedir_knn_lists
  t_file = os.path.join(basedir_knn_lists, "t1/t1_10nn.obj")
  handle = open(t_file, "rb")
  t1_to_10nn = pickle.load(handle)
  handle.close()
  t_file = os.path.join(basedir_knn_lists, "t2/t2_10nn.obj")
  handle = open(t_file, "rb")
  t2_to_10nn = pickle.load(handle)
  handle.close()
  t_file = os.path.join(basedir_knn_lists, "t3/t3_10nn.obj")
  handle = open(t_file, "rb")
  t3_to_10nn = pickle.load(handle)
  handle.close()
  t_file = os.path.join(basedir_knn_lists, "t4/t4_10nn.obj")
  handle = open(t_file, "rb")
  t4_to_10nn = pickle.load(handle)
  handle.close()
  # assert len(t1_to_10nn) == len(t2_to_10nn) and len(t2_to_10nn) == len(t3_to_10nn) and len(t3_to_10nn) == len(t4_to_10nn)
  print('len(t1_to_10nn): %d' % len(t1_to_10nn))
  print('len(t2_to_10nn): %d' % len(t2_to_10nn))
  print('len(t3_to_10nn): %d' % len(t3_to_10nn))
  print('len(t4_to_10nn): %d' % len(t4_to_10nn))

  t_to_10nn_dict = {'t1': t1_to_10nn, 't2': t2_to_10nn, 't3': t3_to_10nn, 't4': t4_to_10nn }
  print('...done.')

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, num_shards, t_to_10nn_dict)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(name, data_dir):
  """Build a list of all images files in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the image data set resides in JPEG files located in
      the following directory structure.

        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg

      where 'dog' is the label associated with these images.

  Returns:
    filenames: list of strings; each string is a path to an image file.
  """
  print('Determining list of input files from %s.' % data_dir)

  filenames = []

  ann_file = FLAGS.train_ann_file if name is 'train' else FLAGS.val_ann_file
  annotations_filepath = os.path.join(data_dir,'annotations',ann_file)
  coco = COCO(annotations_filepath)
  img_ids = coco.getImgIds() # totally 82783 images

  total = 0
  for entry in coco.loadImgs(img_ids):
    total += 1
    if entry['height'] >= FLAGS.image_size and entry['width'] >= FLAGS.image_size:
      ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=None)
      if len(ann_ids) >= FLAGS.min_num_bbox: # len(ann_ids) = #boundingBoxes
        filename = os.path.join(data_dir, 'images', entry['file_name'])
        filenames.append(filename)

  # # Construct the list of JPEG files and labels.
  # jpeg_file_path = '%s/%s/*' % (data_dir, 'images')
  # matching_files = tf.gfile.Glob(jpeg_file_path)
  # filenames.extend(matching_files)

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(4285)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]

  print('Found %d JPEGs inside \'%s\' larger than %s x %s and with at least %s bboxes (of total: %s).' %
        (len(filenames), data_dir, FLAGS.image_size, FLAGS.image_size, FLAGS.min_num_bbox, total))

  if FLAGS.num_images:
    if FLAGS.do_flip:
      num = int(FLAGS.num_images/2)  # div by 2 because of flip
      filenames = filenames[:num]
      print('Reduce number of images to %d (without flip) because of FLAGS.num_images=%d...' % (len(filenames), FLAGS.num_images))
    else:
      num = FLAGS.num_images
      filenames = filenames[:num]
      print('Reduce number of images to %d because of FLAGS.num_images=%d...' % (len(filenames), FLAGS.num_images))

  # print('Found %d JPEG files inside %s.' %
  #       (len(filenames), data_dir))
  return filenames


def _process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
  """
  filenames = _find_image_files(name, directory)
  if len(filenames) == 0:
      return
  _process_image_files(name, filenames, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  # assert not FLAGS.validation_shards % FLAGS.num_threads, (
  #     'Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards')
  print('Saving train results to %s' % FLAGS.train_output_directory)
  print('Saving val results to %s' % FLAGS.val_output_directory)

  # Run it!
  start_time = time.time()
  # _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards)
  duration = round(time.time() - start_time, 2)
  print('duration: ' + str(duration) + 's')


if __name__ == '__main__':
  tf.app.run()
