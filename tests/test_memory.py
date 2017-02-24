import sys, time, json

import numpy as np
import tensorflow as tf

import setup

sys.path.append('model')
from unicycle import Unicycle

sys.path.append('../../tfutils/imagenet/')
from tfutils import model

BATCH_SIZE = 256
MEM = .5
SEED = 12345


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


def test_bypass():
    images = tf.truncated_normal([BATCH_SIZE, 224, 224, 3], seed=SEED)
    labels = tf.constant(range(BATCH_SIZE))

    with tf.variable_scope('unicycle'):
        unicycle_model = Unicycle()
        G = unicycle_model.build(json_file_name='json/sample_alexnet_bypass_test.json')
        out = unicycle_model({'images': images}, G)
        uni_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out.get_output(),
                                                                  labels=labels)

    uni_targets = {'conv1': G.node['conv_1']['tf_cell'].get_output(),
                   'conv2': G.node['conv_2']['tf_cell'].get_output(),
                   'conv3': G.node['conv_3']['tf_cell'].get_output(),
                   'conv4': G.node['conv_4']['tf_cell'].get_output(),
                   'conv5': G.node['conv_5']['tf_cell'].get_output(),
                   'fc6': G.node['fc_6']['tf_cell'].get_output(),
                   'fc7': G.node['fc_7']['tf_cell'].get_output(),
                   'fc8': G.node['fc_8']['tf_cell'].get_output(),
                   'loss': tf.reduce_mean(uni_loss)
                   }
    graph = tf.get_default_graph()
    harbor = graph.get_tensor_by_name('unicycle/conv_1_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 224, 224, 3]
    harbor = graph.get_tensor_by_name('unicycle/conv_2_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 27, 27, 96]
    harbor = graph.get_tensor_by_name('unicycle/conv_3_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 96+256]
    harbor = graph.get_tensor_by_name('unicycle/conv_4_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    harbor = graph.get_tensor_by_name('unicycle/conv_5_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 96+256+256]
    harbor = graph.get_tensor_by_name('unicycle/fc_6_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    harbor = graph.get_tensor_by_name('unicycle/fc_7_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]
    harbor = graph.get_tensor_by_name('unicycle/fc_8_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]


if __name__ == '__main__':
    # mnist = setup.get_mnist_data()
    # test_memory(mnist)

    test_bypass()