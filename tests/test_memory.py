import sys, time

import numpy as np
import tensorflow as tf

import setup

sys.path.append('model')
from unicycle import Unicycle

sys.path.append('../../tfutils/imagenet/')
from tfutils import model

BATCH_SIZE = 256
MEM = .5


def test_memory(mnist):
    images = tf.constant(mnist.train.images[:BATCH_SIZE])

    with tf.variable_scope('unicycle'):
        unicycle_model = Unicycle()
        G = unicycle_model.build(json_file_name='sample_mnist_conv_mem.json')
        unicycle_model({'images': images}, G)

    targets = [G.node['conv_1']['tf_cell'].states,
               G.node['conv_2']['tf_cell'].states,
    ]

    graph = tf.get_default_graph()
    pool1_in = graph.get_tensor_by_name('unicycle/conv_1__f2:0')
    pool2_in = graph.get_tensor_by_name('unicycle/conv_2__f2:0')

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    pool1_state_init = np.zeros(G.node['conv_1']['tf_cell'].get_state().get_shape().as_list())
    pool2_state_init = np.zeros(G.node['conv_2']['tf_cell'].get_state().get_shape().as_list())
    pool1_in, pool2_in = sess.run([pool1_in, pool2_in])

    pool1_state = pool1_state_init.copy()
    pool2_state = pool2_state_init.copy()
    pool1, pool2 = sess.run(targets)
    for p1, p2 in zip(pool1, pool2):
        assert np.allclose(p1, pool1_state)
        pool1_state = pool1_state * MEM + pool1_in
        assert np.allclose(p2, pool2_state)
        pool2_state = pool2_state * MEM + pool2_in

    sess.close()


if __name__ == '__main__':
    mnist = setup.get_mnist_data()
    test_memory(mnist)