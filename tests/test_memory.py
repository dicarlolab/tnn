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

    with tf.variable_scope('unicycle'):
        unicycle_model = Unicycle()
        G = unicycle_model.build(json_file_name='json/sample_alexnet_bypass_test.json')
        out = unicycle_model({'images': images}, G)

    assert G.node['conv_1']['tf_cell'].get_state().shape.as_list() == [BATCH_SIZE, 54, 54, 96]
    assert G.node['conv_2']['tf_cell'].get_state().shape.as_list() == [BATCH_SIZE, 27, 27, 256]
    assert G.node['conv_3']['tf_cell'].get_state().shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    assert G.node['conv_4']['tf_cell'].get_state().shape.as_list() == [BATCH_SIZE, 14, 14, 256]
    assert G.node['conv_5']['tf_cell'].get_state().shape.as_list() == [BATCH_SIZE, 14, 14, 256]
    assert G.node['fc_6']['tf_cell'].get_state().shape.as_list() == [BATCH_SIZE, 4096]
    assert G.node['fc_7']['tf_cell'].get_state().shape.as_list() == [BATCH_SIZE, 4096]
    assert G.node['fc_8']['tf_cell'].get_state().shape.as_list() == [BATCH_SIZE, 1000]

    assert G.node['conv_1']['tf_cell'].get_output().shape.as_list() == [BATCH_SIZE, 27, 27, 96]
    assert G.node['conv_2']['tf_cell'].get_output().shape.as_list() == [BATCH_SIZE, 14, 14, 256]
    assert G.node['conv_3']['tf_cell'].get_output().shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    assert G.node['conv_4']['tf_cell'].get_output().shape.as_list() == [BATCH_SIZE, 14, 14, 256]
    assert G.node['conv_5']['tf_cell'].get_output().shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    assert G.node['fc_6']['tf_cell'].get_output().shape.as_list() == [BATCH_SIZE, 4096]
    assert G.node['fc_7']['tf_cell'].get_output().shape.as_list() == [BATCH_SIZE, 4096]
    assert G.node['fc_8']['tf_cell'].get_output().shape.as_list() == [BATCH_SIZE, 1000]

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
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 96+384+256]
    harbor = graph.get_tensor_by_name('unicycle/fc_6_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    harbor = graph.get_tensor_by_name('unicycle/fc_7_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]
    harbor = graph.get_tensor_by_name('unicycle/fc_8_harbor_concat:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]

    # check if harbor outputs at t are equal to the concat of outputs
    # from incoming nodes at t-1

    # layer 3 gets inputs from 1 and 2
    conv3h = graph.get_tensor_by_name('unicycle/conv_3_harbor_concat_5:0')
    conv1o = G.node['conv_1']['tf_cell'].outputs[4]
    conv1om = tf.nn.max_pool(conv1o, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv2o = G.node['conv_2']['tf_cell'].outputs[4]
    concat = tf.concat([conv1om, conv2o], axis=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv3hr, concatr = sess.run([conv3h, concat])
        assert np.array_equal(conv3hr, concatr)

    # layer 5 gets inputs from 1, 3, 4
    conv5h = graph.get_tensor_by_name('unicycle/conv_5_harbor_concat_7:0')
    conv1o = G.node['conv_1']['tf_cell'].outputs[6]
    conv1om = tf.nn.max_pool(conv1o, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv3o = G.node['conv_3']['tf_cell'].outputs[6]
    conv4o = G.node['conv_4']['tf_cell'].outputs[6]
    # import pdb; pdb.set_trace()
    concat = tf.concat([conv1om, conv3o, conv4o], axis=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv5hr, concatr = sess.run([conv5h, concat])
        assert np.array_equal(conv5hr, concatr)


if __name__ == '__main__':
    # mnist = setup.get_mnist_data()
    # test_memory(mnist)

    test_bypass()