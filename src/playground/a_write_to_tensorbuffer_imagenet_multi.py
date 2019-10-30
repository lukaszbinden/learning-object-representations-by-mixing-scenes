from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys 

import numpy as np
from PIL import Image

from random import shuffle
import cv2
from matplotlib import pyplot as plt
import timeit,time
import tensorflow as tf
import ntpath
from random import randint
import math
import random
import scipy.misc


tf.app.flags.DEFINE_string('directory', '/var/tmp/qhu/weiz_test', 'Place to dump file')
tf.app.flags.DEFINE_string('name', 'weiz_test', 'Name of dump to be produced.')
tf.app.flags.DEFINE_integer('examples_per_file', 5000, 'Number of examples per file. [10000]')
tf.app.flags.DEFINE_integer('image_size', 128, 'Excpected width and length of all images, [227]')

FLAGS = tf.app.flags.FLAGS

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 


def write_to(writer, angle, image,imageR):
    example = tf.train.Example(features=tf.train.Features(feature={
        'angle': _bytes_feature(angle.tostring()),
        'image': _bytes_feature(image.tostring()),
        'imageR': _bytes_feature(imageR.tostring())
    })) 
    writer.write(example.SerializeToString())


def main(argv):

    filename = os.path.join(FLAGS.directory, FLAGS.name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)

    size = [FLAGS.image_size,FLAGS.image_size]
        
    pics = np.zeros([2, size[0]*size[0], 3], dtype=np.uint8)
    pic1 = np.zeros([size[0]*size[1], 3], dtype=np.uint8)
    angles = np.zeros([2,1],dtype=np.float32)
    index = np.zeros([1,1], dtype=np.float32)

    folder_path = 'weizmann_frames'

    idx = 0

    for root, d, files in os.walk(folder_path):
        if files is not None:
              
            for i in xrange(len(files)-1):
                idx = idx+1
                if  idx%6==0:
                    
                    if os.path.exists(os.path.join(root,'frame%d.jpg'%(i))) and os.path.exists(os.path.join(root,'frame%d.jpg'%(i+1))):
                        img1 = scipy.misc.imread(os.path.join(root,'frame%d.jpg'%(i)))
                        img1 = cv2.resize(img1,(size[0],size[1]))
                        pics[0,:] = np.reshape(img1,[size[0]*size[1], 3])
                        b = os.path.join(root,('frame%d.jpg'%(i+1)))

                        img2 = scipy.misc.imread(b)
                        img2 = cv2.resize(img2,(size[0],size[1]))
                        pics[1,:] = np.reshape(img2,[size[0]*size[1], 3])

                        index[0] = np.array(idx)
                        write_to(writer, index[0], pics[0,:,:],pics[1,:]) 

                        # img1 = img2
                        print(idx) 

                        if idx % FLAGS.examples_per_file == 1 and idx >1:
                            filename = os.path.join(FLAGS.directory, FLAGS.name + '.tfrecords'+ str(idx // FLAGS.examples_per_file))
                            writer = tf.python_io.TFRecordWriter(filename)

        print(idx) 

        if idx % FLAGS.examples_per_file == 1 and idx >1:
            filename = os.path.join(FLAGS.directory, FLAGS.name + '.tfrecords'+ str(idx // FLAGS.examples_per_file))
            writer = tf.python_io.TFRecordWriter(filename)
                            
           
        # except Exception,err:
        #     print('Unexcepted image for{} and id is {},rendered_file is {}'.format(rendered_file,idx,type(rendered_file)))
        #     print(Exception, err)
    tf.python_io.TFRecordWriter.close()

if __name__ == '__main__':
    tf.app.run()
