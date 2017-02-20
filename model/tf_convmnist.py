# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def model(data, train=False):
  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()), name='conv1w')
  conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
  conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=0.1,
      seed=SEED, dtype=data_type()), name='conv2w')
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()), name='fc1w')
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()), name='fc2w')
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.

  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  conv1 = tf.nn.conv2d(data,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                       padding='SAME')
  # Bias and rectified linear non-linearity.
  relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
  # Max pooling. The kernel size spec {ksize} also follows the layout of
  # the data. Here we have a pooling window of 2, and a stride of 2.
  pool1 = tf.nn.max_pool(relu1,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  conv2 = tf.nn.conv2d(pool1,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
  pool2 = tf.nn.max_pool(relu2,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')
  # Reshape the feature map cuboid into a 2D matrix to feed it to the
  # fully connected layers.
  pool_shape = pool2.get_shape().as_list()
  reshape = tf.reshape(pool2,
                       [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  # Fully connected layer. Note that the '+' operation automatically
  # broadcasts the biases.
  hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  final = tf.matmul(hidden, fc2_weights) + fc2_biases
  return pool1, pool2, hidden, final
