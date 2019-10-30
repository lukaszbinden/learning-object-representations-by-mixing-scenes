# LZ: taken from for RF study:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/receptive_field/python/util/receptive_field_test.py

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for receptive_fields module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import slim
from tensorflow.contrib.receptive_field import receptive_field_api as receptive_field
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
import tensorflow as tf


# TODO(andrearaujo): Rename the create_test_network_* functions in order to have
# more descriptive names.
def create_test_network_1():
  """Aligned network for test.

  The graph corresponds to the example from the second figure in
  go/cnn-rf-computation#arbitrary-computation-graphs

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = array_ops.placeholder(
        dtypes.float32, (4, 64, 64, 3), name='input_image')
    print('x: ', x.shape)
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    print('l1: ', l1.shape)
    # Right branch.
    l2_pad = array_ops.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    print('l2_pad: ', l2_pad.shape)
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    print('l2: ', l2.shape)
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    print('l3: ', l3.shape)
    # Addition.
    plus = l1 + l3
    print('plus: ', plus.shape)
    nn.relu(plus, name='output')
  return g


def create_test_network_5():
  """Single-path network for testing non-square kernels.
  The graph is similar to the right branch of the graph from
  create_test_network_1(), except that the kernel sizes are changed to be
  non-square.
  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = array_ops.placeholder(
        dtypes.float32, (4, 5, 5, 3), name='input_image')
    # Two convolutional layers, where the first one has non-square kernel.
    l1 = slim.conv2d(x, 1, [3, 3], stride=1, scope='L1', padding='VALID')
    assert l1.shape == (4, 3, 3, 1)
    l2 = slim.conv2d(l1, 1, [3, 3], stride=1, scope='L2', padding='VALID')
    assert l2.shape == (4, 1, 1, 1)
    # ReLU.
    nn.relu(l2, name='output')
  return g


def create_test_network_6():
  """Single-path network for testing non-square kernels.
  The graph is similar to the right branch of the graph from
  create_test_network_1(), except that the kernel sizes are changed to be
  non-square.
  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = array_ops.placeholder(
        dtypes.float32, (4, 5, 5, 3), name='input_image')
    l1 = slim.conv2d(x, 1, [3, 3], stride=2, scope='L1', padding='SAME')
    assert l1.shape == (4, 3, 3, 1)
    l2 = slim.conv2d(l1, 1, [3, 3], stride=2, scope='L2', padding='SAME')
    assert l2.shape == (4, 2, 2, 1)
    l3 = slim.conv2d(l2, 1, [1, 1], stride=1, scope='L3', padding='VALID')
    assert l3.shape == (4, 2, 2, 1)
    # ReLU.
    nn.relu(l3, name='output')
  return g


def create_test_network_7():
  """Single-path network for testing non-square kernels.
  The graph is similar to the right branch of the graph from
  create_test_network_1(), except that the kernel sizes are changed to be
  non-square.
  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = ops.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = array_ops.placeholder(
        dtypes.float32, (4, 64, 64, 3), name='input_image')
    # Two convolutional layers, where the first one has non-square kernel.
    l1 = slim.conv2d(x, 1, [5, 5], stride=2, scope='L1', padding='VALID')
    assert l1.shape == (4, 30, 30, 1)
    l2 = slim.conv2d(l1, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    assert l2.shape == (4, 14, 14, 1)
    # ReLU.
    nn.relu(l2, name='output')
  return g


class ReceptiveFieldTest(test.TestCase):

  def test_create_test_network_6(self):
    graph_def = create_test_network_6().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'

    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))

    self.assertEqual(receptive_field_x, 7)
    self.assertEqual(receptive_field_y, 7)

    print(effective_stride_x, effective_stride_y, effective_padding_x, effective_padding_y)

    # self.assertEqual(effective_stride_x, 1)
    # self.assertEqual(effective_stride_y, 1)
    # self.assertEqual(effective_padding_x, 0)
    # self.assertEqual(effective_padding_y, 0)

  def test_create_test_network_5(self):
    graph_def = create_test_network_5().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'

    res = [n.name for n in graph_def.node]
    print(res)

    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))

    self.assertEqual(receptive_field_x, 5)
    self.assertEqual(receptive_field_y, 5)
    self.assertEqual(effective_stride_x, 1)
    self.assertEqual(effective_stride_y, 1)
    self.assertEqual(effective_padding_x, 0)
    self.assertEqual(effective_padding_y, 0)

  def testComputeRFFromGraphDefAligned(self):
    print("testComputeRFFromGraphDefAligned")

    graph_def = create_test_network_1().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         receptive_field.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node))

    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  # def testComputeRFFromGraphDefAligned2(self):
  #   graph_def = create_test_network_2().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   (receptive_field_x, receptive_field_y, effective_stride_x,
  #    effective_stride_y, effective_padding_x, effective_padding_y) = (
  #        receptive_field.compute_receptive_field_from_graph_def(
  #            graph_def, input_node, output_node))
  #   self.assertEqual(receptive_field_x, 3)
  #   self.assertEqual(receptive_field_y, 3)
  #   self.assertEqual(effective_stride_x, 4)
  #   self.assertEqual(effective_stride_y, 4)
  #   self.assertEqual(effective_padding_x, 1)
  #   self.assertEqual(effective_padding_y, 1)
  #
  # def testComputeRFFromGraphDefUnaligned(self):
  #   graph_def = create_test_network_3().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   with self.assertRaises(ValueError):
  #     receptive_field.compute_receptive_field_from_graph_def(
  #         graph_def, input_node, output_node)
  #
  # def testComputeRFFromGraphDefUndefinedPadding(self):
  #   graph_def = create_test_network_4().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   (receptive_field_x, receptive_field_y, effective_stride_x,
  #    effective_stride_y, effective_padding_x, effective_padding_y) = (
  #        receptive_field.compute_receptive_field_from_graph_def(
  #            graph_def, input_node, output_node))
  #   self.assertEqual(receptive_field_x, 3)
  #   self.assertEqual(receptive_field_y, 3)
  #   self.assertEqual(effective_stride_x, 4)
  #   self.assertEqual(effective_stride_y, 4)
  #   self.assertEqual(effective_padding_x, None)
  #   self.assertEqual(effective_padding_y, None)
  #
  # def testComputeRFFromGraphDefFixedInputDim(self):
  #   graph_def = create_test_network_4().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   (receptive_field_x, receptive_field_y, effective_stride_x,
  #    effective_stride_y, effective_padding_x, effective_padding_y) = (
  #        receptive_field.compute_receptive_field_from_graph_def(
  #            graph_def, input_node, output_node, input_resolution=[9, 9]))
  #   self.assertEqual(receptive_field_x, 3)
  #   self.assertEqual(receptive_field_y, 3)
  #   self.assertEqual(effective_stride_x, 4)
  #   self.assertEqual(effective_stride_y, 4)
  #   self.assertEqual(effective_padding_x, 1)
  #   self.assertEqual(effective_padding_y, 1)
  #
  # def testComputeRFFromGraphDefUnalignedFixedInputDim(self):
  #   graph_def = create_test_network_4().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   with self.assertRaises(ValueError):
  #     receptive_field.compute_receptive_field_from_graph_def(
  #         graph_def, input_node, output_node, input_resolution=[8, 8])
  #
  # def testComputeRFFromGraphDefNonSquareRF(self):
  #   graph_def = create_test_network_5().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   (receptive_field_x, receptive_field_y, effective_stride_x,
  #    effective_stride_y, effective_padding_x, effective_padding_y) = (
  #        receptive_field.compute_receptive_field_from_graph_def(
  #            graph_def, input_node, output_node))
  #   self.assertEqual(receptive_field_x, 5)
  #   self.assertEqual(receptive_field_y, 7)
  #   self.assertEqual(effective_stride_x, 4)
  #   self.assertEqual(effective_stride_y, 4)
  #   self.assertEqual(effective_padding_x, 0)
  #   self.assertEqual(effective_padding_y, 0)
  #
  # def testComputeRFFromGraphDefStopPropagation(self):
  #   graph_def = create_test_network_6().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   # Compute the receptive field but stop the propagation for the random
  #   # uniform variable of the dropout.
  #   (receptive_field_x, receptive_field_y, effective_stride_x,
  #    effective_stride_y, effective_padding_x, effective_padding_y) = (
  #        receptive_field.compute_receptive_field_from_graph_def(
  #            graph_def, input_node, output_node,
  #            ['Dropout/dropout/random_uniform']))
  #   self.assertEqual(receptive_field_x, 3)
  #   self.assertEqual(receptive_field_y, 3)
  #   self.assertEqual(effective_stride_x, 4)
  #   self.assertEqual(effective_stride_y, 4)
  #   self.assertEqual(effective_padding_x, 1)
  #   self.assertEqual(effective_padding_y, 1)
  #
  # def testComputeCoordinatesRoundtrip(self):
  #   graph_def = create_test_network_1()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   rf = receptive_field.compute_receptive_field_from_graph_def(
  #       graph_def, input_node, output_node)
  #
  #   x = np.random.randint(0, 100, (50, 2))
  #   y = rf.compute_feature_coordinates(x)
  #   x2 = rf.compute_input_center_coordinates(y)
  #
  #   self.assertAllEqual(x, x2)
  #
  # def testComputeRFFromGraphDefAlignedWithControlDependencies(self):
  #   graph_def = create_test_network_7().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   (receptive_field_x, receptive_field_y, effective_stride_x,
  #    effective_stride_y, effective_padding_x, effective_padding_y) = (
  #        receptive_field.compute_receptive_field_from_graph_def(
  #            graph_def, input_node, output_node))
  #   self.assertEqual(receptive_field_x, 3)
  #   self.assertEqual(receptive_field_y, 3)
  #   self.assertEqual(effective_stride_x, 4)
  #   self.assertEqual(effective_stride_y, 4)
  #   self.assertEqual(effective_padding_x, 1)
  #   self.assertEqual(effective_padding_y, 1)
  #
  # def testComputeRFFromGraphDefWithIntermediateAddNode(self):
  #   graph_def = create_test_network_8().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   (receptive_field_x, receptive_field_y, effective_stride_x,
  #    effective_stride_y, effective_padding_x, effective_padding_y) = (
  #        receptive_field.compute_receptive_field_from_graph_def(
  #            graph_def, input_node, output_node))
  #   self.assertEqual(receptive_field_x, 11)
  #   self.assertEqual(receptive_field_y, 11)
  #   self.assertEqual(effective_stride_x, 8)
  #   self.assertEqual(effective_stride_y, 8)
  #   self.assertEqual(effective_padding_x, 5)
  #   self.assertEqual(effective_padding_y, 5)
  #
  # def testComputeRFFromGraphDefWithIntermediateAddNodeSamePaddingFixedInputDim(
  #     self):
  #   graph_def = create_test_network_9().as_graph_def()
  #   input_node = 'input_image'
  #   output_node = 'output'
  #   (receptive_field_x, receptive_field_y, effective_stride_x,
  #    effective_stride_y, effective_padding_x, effective_padding_y) = (
  #        receptive_field.compute_receptive_field_from_graph_def(
  #            graph_def, input_node, output_node, input_resolution=[17, 17]))
  #   self.assertEqual(receptive_field_x, 11)
  #   self.assertEqual(receptive_field_y, 11)
  #   self.assertEqual(effective_stride_x, 8)
  #   self.assertEqual(effective_stride_y, 8)
  #   self.assertEqual(effective_padding_x, 5)
  #   self.assertEqual(effective_padding_y, 5)


if __name__ == '__main__':
  test.main()