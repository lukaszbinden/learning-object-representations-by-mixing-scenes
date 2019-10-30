r"""
Source: https://github.com/MetaPeak/tensorflow_object_detection_create_coco_tfrecord
Modified by Lukas Zbinden (image resizing).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
import os
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/data/cvg/lukas/datasets/coco', 'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set or validation set')
FLAGS = flags.FLAGS


def create_for_coco_dataset(imgs_dir, annotations_filepath):
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

    total = 0
    greaterThan300 = 0
    moreThan4 = 0
    for entry in coco.loadImgs(img_ids):
        total += 1
        if entry['height'] >= 300 and entry['width'] >= 300:
            greaterThan300 += 1
            ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=None)
            if len(ann_ids) > 4:
                moreThan4 += 1

    print('total: %d' % total)
    print('greaterThan300: %d' % greaterThan300)
    print('moreThan4: %d' % moreThan4)

def create_session():
    sess = tf.Session()
    return sess

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

    create_for_coco_dataset(imgs_dir, annotations_filepath)


if __name__ == "__main__":
    tf.app.run()

