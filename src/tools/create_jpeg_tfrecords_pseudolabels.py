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

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels] where 0 is unused and left as
    the background class.
  image/class/text: string specifying the human-readable version of the label
    e.g. 'dog'

If your data set involves bounding boxes, please look at build_imagenet_data.py.

LZ: modified
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

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
import traceback
from scipy.misc import imsave
from data_augment import ImageCoder, augment_image
import pickle

tf.app.flags.DEFINE_string('train_directory', '/home/lz01a008/src/datasets/coco/2017_training/version/v5/full/',
                           'Training data directory')
tf.app.flags.DEFINE_string('filename_pseudolabel_directory', '/home/lz01a008/src/logs/20190511_161149/',
                           'directory where fname_pseudolabel.obj resides')
tf.app.flags.DEFINE_string('train_ann_file', 'instances_train2017.json',
                           'Training data annotation file')
tf.app.flags.DEFINE_string('validation_directory', '/data/cvg/lukas/datasets/coco/2017_test/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('val_ann_file', 'instances_val2017.json',
                           'Validation data annotation file')
tf.app.flags.DEFINE_string('train_output_directory', '/home/lz01a008/src/datasets/coco/2017_training/version/v7/final',
                           'Train Output data directory')
tf.app.flags.DEFINE_string('val_output_directory', '/data/cvg/lukas/datasets/coco/2017_test/version/v2/tmp',
                           'Validation Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 150,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 10,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 10,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('image_size', 200,
                            'Excpected width and length of all images, [300]')
tf.app.flags.DEFINE_integer('min_num_bbox', 0,
                            'Minimum number of bounding boxes / objects, [5]')
tf.app.flags.DEFINE_integer('num_crops', 4,
                            'Number of crops per image, [3]')
tf.app.flags.DEFINE_integer('num_images', None,
                            'Number of images to use (incl. flips), None -> all')
tf.app.flags.DEFINE_integer('target_image_size', 224,
                            'The target image size for scaled and randomly cropped images')
tf.app.flags.DEFINE_boolean('data_augmentation', False,
                            'Apply data augmentation incl. flip, scale, random crop, [True]')
tf.app.flags.DEFINE_boolean('dump_images', False,
                            'Dump images to *_output_directory if True')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, height, width, label):
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
  assert type(label) == int

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/label': _int64_feature(label),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example


def _process_image(filename_pslabel, coder):
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

  file_path = os.path.join(FLAGS.train_directory, filename_pslabel[0])
  with tf.gfile.FastGFile(file_path, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  # del image_data

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3, 'image %s not RGB but %d' % (filename_pslabel[0], image.shape[2])


  # return augment_image(image_data, image, height, width, coder, FLAGS.data_augmentation)
  return image_data, height, width, image, filename_pslabel[1]


# def random_crop_max(coder, image, result, heights, widths, images, height, width):
#     rc2 = coder.random_crop_max(image, height, width)
#     image_data = coder.encode_jpeg(rc2)
#     result.append(image_data)
#     heights.append(rc2.shape[0])
#     widths.append(rc2.shape[1])
#     images.append(rc2)
#
#
# def scale_image(coder, result, heights, widths, images, height, image, width, factors):
#   for factor in factors: # add more factors if required
#     cropped = coder.scale(image, height, width, factor)
#     image_data = coder.encode_jpeg(cropped)
#     result.append(image_data)
#     heights.append(cropped.shape[0])
#     widths.append(cropped.shape[1])
#     images.append(cropped)


def _process_image_files_batch(coder, thread_index, ranges, name, filenames_pslabels, num_shards):
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
  # num_files_in_thread *= FLAGS.num_crops

  counter = 0
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
      filename_pslabel = filenames_pslabels[i]

      try:
        image_data, height, width, image, label = _process_image(filename_pslabel, coder)
      except Exception as e:
        print(e)
        tb = traceback.format_exc()
        print(tb)
        print('SKIPPED: Unexpected error while decoding %s.' % filename_pslabel)
        continue

      image_buffer = image_data
      fn = filename_pslabel[0]
      example = _convert_to_example(fn, image_buffer, height, width, label)
      writer.write(example.SerializeToString())

    if FLAGS.dump_images:
        assert 1 == 0, "not supported!"

    shard_counter += 1
    counter += 1

    if not counter % 1000:
      print('%s [thread %d]: Processed %d of %d images in thread batch.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
      sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
  print('%s [thread %d]: Wrote %d images from %d files.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames_pslabel, num_shards):
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames_pslabel), FLAGS.num_threads + 1).astype(np.int)
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

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames_pslabel, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames_pslabel)))
  sys.stdout.flush()


def _find_image_files(_, data_dir):

  fn_pl_path = os.path.join(data_dir, "fname_pseudolabel.obj")
  print('Determining list of input files from %s.' % fn_pl_path)
  handle = open(fn_pl_path, "rb")
  fn_psl = pickle.load(handle)
  handle.close()

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.

  shuffled_index = list(range(len(fn_psl)))
  random.seed(4285)
  random.shuffle(shuffled_index)

  filenames_pslabel = [fn_psl[i] for i in shuffled_index]

  if FLAGS.num_images:
    assert 1 == 0, "not supported!"

  print('Found %d JPEG files inside %s.' % (len(filenames_pslabel), fn_pl_path))
  return filenames_pslabel


def _process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
  """
  filenames_pslabel = _find_image_files(name, directory)
  if len(filenames_pslabel) == 0:
      return
  _process_image_files(name, filenames_pslabel, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  # assert not FLAGS.validation_shards % FLAGS.num_threads, (
  #     'Please make the FLAGS.num_threads commensurate with FLAGS.validation_shards')
  print('Saving train results to %s' % FLAGS.train_output_directory)

  # Run it!
  start_time = time.time()
  # _process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards)
  _process_dataset('train', FLAGS.filename_pseudolabel_directory, FLAGS.train_shards)
  duration = round(time.time() - start_time, 2)
  print('duration: ' + str(duration) + 's')


if __name__ == '__main__':
  tf.app.run()

