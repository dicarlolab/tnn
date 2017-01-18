"""
The "model" function is used to create a TF graph given layers and bypasses

The model function imports supplementary functions from other files, one 
function type per file for the most part

Let's do this
"""

from __future__ import absolute_import, division, print_function

import networkx as nx
import tensorflow as tf
from ConvRNNCell import ConvRNNCell
from alexnet import alexnet

from constructor import _construct_graph
from support_functions import _first, _last, _maxpool, get_loss

from complete_graph import _complete_graph



def get_model(inputs,
              train=False,
              cfg_initial=None,
              seed=None,
              model_base=None,
              bypasses=[], 
              init_weights='xavier',
              weight_decay=None,
              dropout=None,
              memory_decay=None,
              memory_trainable=False,
              trim_top=True,
              trim_bottom=True,
              features_layer=None,
              bypass_pool_kernel_size=None,
              input_spatial_size=None,
              input_seq_len=1,
              target='data'):
    """
    Creates model graph and returns logits.

    :inputs: list for sequence of input images as tf Tensors

    :train: specify if the current instance uses training or evaluation only
    :cfg_initial:
    :seed:
    :model_base:
    :bypasses:
    :init_weights='xavier':
    :weight_decay=None:
    :dropout=None:
    :memory_decay=None:
    :memory_trainable=False:
    :trim_top=True:
    :trim_bottom=True:
    :features_layer: if None (equivalent to a value of len(layers) + 1) ,
                     outputs logitsfrom last FC. Otherwise, accepts a number 0 
                     through len(layers) + 1 and _model will output the features 
                     of that layer.
    :bypass_pool_kernel_size=None:
    :input_spatial_size=None:
    :input_seq_len=1:
    :target='data':


    model_base: string name of model base. (Ex: 'alexnet')
    :param layers: Dictionary to construct cells for each layer of the form
     {layer #: ['cell type', {arguments}] Does not include the final linear
     layer used to get logits.
    :param bypasses: list of tuples (from, to)
    :param inputs: list for sequence of input images as tf Tensors
    :param initial_states: optional; dict of initial state {layer#: tf Tensor}
    :return: Returns a dictionary logits (output of a linear FC layer after
    all layers). {time t: logits} for t >= shortest_path and t < T_total}
    """

    # create networkx graph with layer #s as nodes
    if model_base is None: model_base = alexnet
    layers = model_base(input_spatial_size=input_spatial_size,
                        batch_size=inputs[target].get_shape().as_list()[0],
                        init_weights=init_weights,
                        weight_decay=weight_decay,
                        memory_decay=memory_decay,
                        memory_trainable=memory_trainable,
                        dropout=dropout,
                        train=train,
                        seed=seed)
    graph = _construct_graph(layers, bypasses)

    nlayers = len(layers)  # number of layers including the final FC/logits
    shortest_path = nx.shortest_path_length(graph, source='0',
                                            target=str(nlayers))
    ntimes = input_seq_len + shortest_path - 1  # total num. of time steps

    # ensure that graph is acyclic
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('graph not acyclic')

    # get first and last time points where layers matter
    if trim_top:
        _first(graph)
    else:
        graph.node['0']['first'] = 0  # input matters at t = 0,
        # rest starting t = 1
        for node in graph:
            graph.node[node]['first'] = 1

    if trim_bottom:
        _last(graph, ntimes)
    else:
        for node in graph:
            graph.node[node]['last'] = ntimes

    # # check inputs: Compares input sequence length with the input length
    # # that is needed for output at T_tot. Zero pads or truncates as needed
    # # TODO: can this scenario happen?
    # if len(input_seq) > graph.node['0']['last'] + 1:  # more inputs than needed => truncate
    #     print('truncating input sequence to length', graph.node['0']['last'] + 1)
    #     del input_seq[graph.node['0']['last'] + 1:]
    # elif len(input_seq) < graph.node['0']['last'] + 1:  # too short => pad with zero inputs
    #     print('zero-padding input sequence to length', graph.node['0']['last'] + 1)
    #     num_needed = (graph.node['0']['last'] + 1) - len(input_seq)  # inputs to add
    #     if not input_seq:  # need input length of at least one
    #         raise ValueError('input sequence should not be empty')
    #     padding = [{'data': tf.zeros_like(input_seq[0]['data'])}
    #                for i in range(0, num_needed)]
    #     input_seq.extend(padding)

    # add inputs to outputs dict for layer 0
    # outputs = {layer#: {t1:__, t2:__, ... }}
    graph.node['0']['inputs'] = None
    graph.node['0']['outputs'] = [inputs[target] for _ in range(input_seq_len)]
    graph.node['0']['initial_states'] = None
    graph.node['0']['final_states'] = None

    # create zero initial states if none specified
    # if initial_states is None:
    # zero state returns zeros (tf.float32) based on state size.
    for layer in graph:
        if layer == '0':
            st = None
        else:
            st = graph.node[layer]['cell'].zero_state(None, None)
        graph.node[layer]['initial_states'] = st

    reuse = None if train else True
    # create graph layer by layer
    for n, node in enumerate(sorted(graph.nodes())):
        if node != '0':
            layer = graph.node[node]
            with tf.variable_scope(layer['name'], reuse=reuse):
                import pdb; pdb.set_trace()
                # print('{:-^80}'.format(layer['name']))
                # create inputs list for layer j, each element is an input in time
                layer['inputs'] = []  # list of inputs to layer j in time
                parents = graph.predecessors(node)  # list of incoming nodes

                ###
                ### COMPLETE GRAPH GOES HERE
                ###

                # run tf.nn.rnn and get list of outputs
                # Even if initial_states[j] is None, tf.nn.rnn will just set
                # zero initial state (given dtype)
                if len(layer['inputs']) > 0:
                    out, fstate = tf.nn.rnn(cell=layer['cell'],
                                            inputs=layer['inputs'],
                                            initial_state=layer['initial_states'],
                                            dtype=tf.float32)
                else:  # if empty, layer j doesn't contribute to output t<= T_tot
                    fstate = layer['initial_states']

                # fill in empty outputs with zeros since we index by t
                out_first = []
                for t in range(0, layer['first']):
                    out_first.append(
                        tf.zeros(shape=layer['cell'].output_size,
                                 dtype=tf.float32))
                out = out_first + out

                layer['outputs'] = out
                layer['final_states'] = fstate

    for node in graph:
        if node != '0':
            layer = graph.node[node]
            layer['outputs'] = layer['outputs'][layer['first']: layer['last'] + 1]

    if features_layer is None:
        return graph.node[str(len(layers))]['outputs'], cfg_initial
    else:
        for node in graph:
            if graph.node[node]['name'] == features_layer:
                return graph.node[node]['outputs'], cfg_initial
                break
        else:
            raise ValueError('Layer {} not found'.format(features_layer))


