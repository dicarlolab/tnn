import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('model')
from unicycle import Unicycle


# Import MNIST data
BATCH_SIZE = 256
SEED = 12345


def get_mnist_data():
    # mnist variable is an instance of the DataSet class.
    mnist = input_data.read_data_sets('MNIST_data/')
    assert mnist.train.images.shape == (55000, 784)
    assert mnist.train.labels.shape == (55000,)
    assert mnist.test.images.shape == (10000, 784)
    assert mnist.test.labels.shape == (10000,)
    assert mnist.validation.images.shape == (5000, 784)
    assert mnist.validation.labels.shape == (5000,)
    return mnist


def mnist_fc(images, labels):
    init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
    weights_fc_1 = tf.get_variable(name='weights_fc_1', shape=[784, 2048], initializer=init)
    init = tf.constant_initializer(.1)
    biases_fc_1 = tf.get_variable('biases_fc_1', shape=[2048], initializer=init)
    fc1 = tf.nn.relu(tf.matmul(images, weights_fc_1) + biases_fc_1)

    init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
    weights_fc_2 = tf.get_variable(name='weights_fc_2', shape=[2048, 10], initializer=init)
    init = tf.constant_initializer(.1)
    biases_fc_2 = tf.get_variable('biases_fc_2', shape=[10], initializer=init)
    fc2 = tf.matmul(fc1, weights_fc_2) + biases_fc_2

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=labels)
    return {'fc1': fc1, 'fc2': fc2, 'loss': tf.reduce_mean(loss)}


def mnist_conv(images, labels):
    images = tf.reshape(images, [-1, 28, 28, 1])

    init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
    weights_conv_1 = tf.get_variable(name='weights_conv_1', shape=[5, 5, 1, 32], initializer=init)
    init = tf.constant_initializer(0)
    biases_conv_1 = tf.get_variable('biases_conv_1', shape=[32], initializer=init)
    conv1 = tf.nn.relu(tf.nn.conv2d(images, weights_conv_1,
                                    strides=[1, 1, 1, 1], padding='SAME') + biases_conv_1)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
    weights_conv_2 = tf.get_variable(name='weights_conv_2', shape=[5, 5, 32, 64], initializer=init)
    init = tf.constant_initializer(.1)
    biases_conv_2 = tf.get_variable('biases_conv_2', shape=[64], initializer=init)
    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights_conv_2, strides=[1, 1, 1, 1], padding='SAME') + biases_conv_2)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2_resh = tf.reshape(pool2, [pool2.get_shape().as_list()[0], -1])

    init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
    weights_fc_1 = tf.get_variable(name='weights_fc_1',
                                   shape=[pool2_resh.get_shape().as_list()[-1], 512],
                                   initializer=init)
    init = tf.constant_initializer(.1)
    biases_fc_1 = tf.get_variable('biases_fc_1', shape=[512], initializer=init)
    fc1 = tf.nn.relu(tf.matmul(pool2_resh, weights_fc_1) + biases_fc_1)

    init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
    weights_fc_2 = tf.get_variable(name='weights_fc_2', shape=[512, 10], initializer=init)
    init = tf.constant_initializer(.1)
    biases_fc_2 = tf.get_variable('biases_fc_2', shape=[10], initializer=init)
    fc2 = tf.matmul(fc1, weights_fc_2) + biases_fc_2

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=labels)
    return {'pool1': pool1, 'pool2': pool2, 'fc1': fc1, 'fc2': fc2, 'loss': loss}


def test_mnist_fc(mnist):

    # initialize the benchmark model
    with tf.variable_scope('benchmark'):
        bench_targets = mnist_fc(mnist.train.images[:BATCH_SIZE],
                                 mnist.train.labels[:BATCH_SIZE].astype(np.int32))

    bench_vars = {v.name.split('/')[1]:v for v in tf.global_variables() if v.name.startswith('benchmark')}
    bench_targets.update(bench_vars)
    for name, var in bench_vars.items():
        bench_targets['grad_' + name] = tf.gradients(bench_targets['loss'], var)

    # initialize the unicycle model
    with tf.variable_scope('unicycle'):
        unicycle_model = Unicycle()
        G = unicycle_model.build(json_file_name='sample_mnist_v2.json')
        out = unicycle_model({'images': tf.constant(mnist.train.images[:BATCH_SIZE])}, G)
        labels = mnist.train.labels[:BATCH_SIZE].astype(np.int32)
        uni_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out.get_output(), labels=labels)

    uni_targets = {'fc1': G.node['fc_1']['tf_cell'].get_output(),
                   'fc2': G.node['fc_2']['tf_cell'].get_output(),
                   'loss': tf.reduce_mean(uni_loss)}
    uni_vars = {v.name.split('/')[1]:v for v in tf.global_variables()
                if v.name.startswith('unicycle') and 'decay_param' not in v.name}
    uni_targets.update(uni_vars)
    for name, var in uni_vars.items():
        uni_targets['grad_' + name] = tf.gradients(uni_targets['loss'], var)

    run(bench_targets, uni_targets, nsteps=100)


def test_mnist_conv(mnist):
    # initialize the benchmark model
    with tf.variable_scope('benchmark'):
        bench_targets = mnist_conv(mnist.train.images[:BATCH_SIZE],
                                   mnist.train.labels[:BATCH_SIZE].astype(np.int32))

    bench_vars = {v.name.split('/')[1]:v for v in tf.global_variables()
                 if v.name.startswith('benchmark') and not v.name.startswith('pool')}
    bench_targets.update(bench_vars)
    for name, var in bench_vars.items():
        bench_targets['grad_' + name] = tf.gradients(bench_targets['loss'], var)


    # initialize the unicycle model
    with tf.variable_scope('unicycle'):
        unicycle_model = Unicycle()
        G = unicycle_model.build(json_file_name='sample_mnist_conv.json')
        out = unicycle_model({'images': tf.constant(mnist.train.images[:BATCH_SIZE])}, G)
        labels = mnist.train.labels[:BATCH_SIZE].astype(np.int32)
        uni_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out.get_output(), labels=labels)

    uni_targets = {'pool1': G.node['conv_1']['tf_cell'].get_output(),
                   'pool2': G.node['conv_2']['tf_cell'].get_output(),
                   'fc1': G.node['fc_1']['tf_cell'].get_output(),
                   'fc2': G.node['fc_2']['tf_cell'].get_output(),
                   'loss': uni_loss}
    uni_vars = {v.name.split('/')[1]:v for v in tf.global_variables()
                 if v.name.startswith('unicycle') and 'decay_param' not in v.name}

    uni_targets.update(uni_vars)
    for name, var in uni_vars.items():
        uni_targets['grad_' + name] = tf.gradients(uni_targets['loss'], var)

    run(bench_targets, uni_targets, nsteps=100)


def run(bench_targets, uni_targets, nsteps=100):
    assert np.array_equal(sorted(uni_targets.keys()), sorted(bench_targets.keys()))

    opt = tf.train.MomentumOptimizer(learning_rate=.01, momentum=.9)
    bench_targets['optimizer'] = opt.minimize(bench_targets['loss'])

    opt = tf.train.MomentumOptimizer(learning_rate=.01, momentum=.9)
    uni_targets['optimizer'] = opt.minimize(uni_targets['loss'])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for step in range(nsteps):
        # check if the outputs are identical
        if step < 4:
            bench_res = sess.run(bench_targets)
            uni_res = sess.run(uni_targets)

            for name in bench_res:
                if name != 'optimizer':
                    assert np.allclose(bench_res[name], uni_res[name], atol=1e-2, rtol=1e-2)

        else:  # after that, stuff starts to diverge too much, but the loss should be ok
            bench_loss = sess.run(bench_targets['loss'])
            uni_loss = sess.run(uni_targets['loss'])
            assert np.allclose(bench_loss, uni_loss, atol=1e-2, rtol=1e-2)

    sess.close()


if __name__ == '__main__':
    mnist = get_mnist_data()
    test_mnist_fc(mnist)
    tf.reset_default_graph()
    test_mnist_conv(mnist)

