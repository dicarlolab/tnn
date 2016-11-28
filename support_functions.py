"""
Additional functions to help with the logistics of unrolling, pooling and evaluation
"""

from __future__ import absolute_import, division, print_function

import networkx as nx
import tensorflow as tf

def _first(graph):
    """
    Returns dictionary with times of when each layer first matters, that is,
    receives information from input layer.
    :param graph: networkx graph representing model
    :return: dictionary first[j] = time t where layer j matters. j ranges
    from 0 to N_cells + 1
    """
    curr_layers = ['0']
    t = 0
    while len(curr_layers) > 0:
        next_layers = []
        for layer in curr_layers:
            if 'first' not in graph.node[layer]:
                graph.node[layer]['first'] = t
                next_layers.extend(graph.successors(layer))
        curr_layers = next_layers
        t += 1


def _last(graph, ntimes):
    """
    Returns dictionary with times of when each layer last matters, that is,
    last time t where layer j information reaches output layer at, before T_tot
    Note last[0] >= 0, for input layer
    :param graph: networkx graph representing model
    :param ntimes: total number of time steps to run the model.
    :return: dictionary {layer j: last time t}
    """
    curr_layers = [str(len(graph) - 1)]  # start with output layer
    t = ntimes
    while len(curr_layers) > 0:
        next_layers = []  # layers at prev time point
        for layer in curr_layers:
            if 'last' not in graph.node[layer]:
                graph.node[layer]['last'] = t
                # then add adjacency list onto next_layer
                next_layers.extend(graph.predecessors(layer))
        curr_layers = next_layers
        t -= 1


def _maxpool(input_, out_spatial, kernel_size=None, name='pool'):
    """
    Returns a tf operation for maxpool of input
    Stride determined by the spatial size ratio of output and input
    kernel_size = None will set kernel_size same as stride.
    """
    in_spatial = input_.get_shape().as_list()[1]
    stride = in_spatial // out_spatial  # how much to pool by
    if stride < 1:
        raise ValueError('spatial dimension of output should not be greater '
                         'than that of input')
    if kernel_size is None:
        kernel_size = stride
    pool = tf.nn.max_pool(input_,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1],
                          padding='VALID',
                          name=name)
    return pool


def get_loss(inputs,
             outputs,
             target,
             loss_per_case_func,
             agg_func,
             loss_func_kwargs=None,
             agg_func_kwargs=None,
             time_penalty=1.2):
    if loss_func_kwargs is None:
        loss_func_kwargs = {}
    if agg_func_kwargs is None:
        agg_func_kwargs = {}

    losses = []
    for t, out in enumerate(outputs):
        loss_t = loss_per_case_func(out, inputs[target],
                                    name='xentropy_loss_t{}'.format(t),
                                    **loss_func_kwargs)
        loss_t_mean = agg_func(loss_t, **agg_func_kwargs)
        loss_t_mean *= time_penalty**t
        losses.append(loss_t_mean)
    # use 'losses' collection to also add weight decay loss
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = sum(losses) + sum(reg_losses)

    return total_loss