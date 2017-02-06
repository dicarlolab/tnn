"""
Unicycle Utility Functions
"""

from math import floor, ceil
from random import randint
import tensorflow as tf


def create_kwargs_conv(fn, input_size, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict = {}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type'] == 'conv', 'Type has to be CONV, but is %s' % \
        (fn['type'])
    # filter_tensor = \
    #     tf.get_variable('filter_tensor_%s' % (name),
    #                     fn['filter_size']
    #                     + [input_size[-1]]
    #                     + [fn['num_filters']],
    #                     initializer=initializer()
    #                     )
    # out_dict['filter'] = filter_tensor
    out_dict['out_shape'] = fn['num_filters']
    out_dict['ksize'] = fn['filter_size']
    out_dict['stride'] = fn['stride']
    if 'padding' in fn:
        out_dict['padding'] = fn['padding'].upper()
    if 'init' in fn:
        out_dict['init'] = initializer(
            kind=fn['init'],
            stddev=fn['stddev'] if 'stddev' in fn else .01)
    if 'bias' in fn:
        out_dict['bias'] = fn['bias']
    out_dict['name'] = str(name) if isinstance(name, basestring) \
        else 'conv_%s' % (str(randint(1, 1000)))

    return out_dict, calc_size_after(input_size, fn)


def create_kwargs_maxpool(fn, input_size, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict = {}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type'] == 'maxpool', 'Type has to be MAXPOOL, but is %s' % \
        (fn['type'])

    out_dict['ksize'] = [1, fn['k_size'], fn['k_size'], 1]
    out_dict['strides'] = [1, fn['stride'], fn['stride'], 1]
    out_dict['padding'] = fn['padding'].upper()
    out_dict['name'] = str(name) if isinstance(name, basestring) \
        else 'maxpool_%s' % (str(randint(1, 1000)))
    return out_dict, calc_size_after(input_size, fn)


def create_kwargs_relu(fn, input_size, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict = {}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type'] == 'relu', 'Type has to be RELU, but is %s' % \
        (fn['type'])

    out_dict['name'] = str(name) if isinstance(name, basestring) \
        else 'relu_%s' % (str(randint(1, 1000)))
    return out_dict, calc_size_after(input_size, fn)


def create_kwargs_norm(fn, input_size, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict = {}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type'] == 'norm', 'Type has to be NORM, but is %s' % \
        (fn['type'])

    out_dict['depth_radius'] = fn['depth_radius']
    out_dict['bias'] = fn['bias']
    out_dict['alpha'] = fn['alpha']
    out_dict['name'] = str(name) if isinstance(name, basestring) \
        else 'norm_%s' % (str(randint(1, 1000)))
    return out_dict, calc_size_after(input_size, fn)


def create_kwargs_fc(fn, input_size):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict = {}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type'] == 'fc', 'Type has to be FC, but is %s' % (fn['type'])

    out_dict['output_size'] = fn['output_size']
    if 'init' in fn:
        init = initializer(kind=fn['init'])
    else:
        init = initializer()
    out_dict['init'] = init
    if 'bias' in fn:
        out_dict['bias'] = fn['bias']
    return out_dict, calc_size_after(input_size, fn)


def create_kwargs_ph(fn, input_size):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict = {}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type'] == 'placeholder', 'Type has to be PLACEHOLDER, but is \
                                                        %s' % (fn['type'])

    out_dict['shape'] = input_size
    return out_dict, calc_size_after(input_size, fn)


def assemble_function_kwargs(functions, input_size, nickname):
    to_be_passed_in_state_kwargs = []
    cur_size = input_size[:]
    for f in functions:
        f_type = f['type']
        temp_kwarg = {}
        if f_type == 'conv':
            temp_kwarg, cur_size = create_kwargs_conv(f, cur_size, nickname)
        elif f_type == 'maxpool':
            temp_kwarg, cur_size = create_kwargs_maxpool(f, cur_size, nickname)
        elif f_type == 'relu':
            temp_kwarg, cur_size = create_kwargs_relu(f, cur_size, nickname)
        elif f_type == 'norm':
            temp_kwarg, cur_size = create_kwargs_norm(f, cur_size, nickname)
        elif f_type == 'fc':
            temp_kwarg, cur_size = create_kwargs_fc(f, cur_size)
        elif f_type == 'placeholder':
            temp_kwarg, cur_size = create_kwargs_ph(f, input_size)
        to_be_passed_in_state_kwargs.append(temp_kwarg)
    return to_be_passed_in_state_kwargs


def calc_size_after(input_size, function_):
    print input_size
    if function_['type'] == 'conv':
        if function_['padding'] == 'same':
            out_height = int(ceil(float(input_size[1])
                                  / float(function_['stride'])))
            out_width = int(ceil(float(input_size[2])
                                 / float(function_['stride'])))
        elif function_['padding'] == 'valid':
            out_height = int(ceil(float(input_size[1]
                                        - function_['filter_size'] + 1)
                                  / float(function_['stride'])))
            out_width = int(ceil(float(input_size[2]
                                       - function_['filter_size'] + 1)
                                 / float(function_['stride'])))
        return [input_size[0], out_height, out_width, function_['num_filters']]

    elif function_['type'] == 'maxpool':
        if function_['padding'] == 'valid':
            out_height = floor(float(input_size[1] - function_['k_size'])
                               / float(function_['stride'])) + 1
            out_width = floor(float(input_size[2] - function_['k_size'])
                              / float(function_['stride'])) + 1
        elif function_['padding'] == 'same':
            out_height = ceil(float(input_size[1] - function_['k_size'])
                              / float(function_['stride'])) + 1
            out_width = ceil(float(input_size[2] - function_['k_size'])
                             / float(function_['stride'])) + 1
        return [input_size[0], int(out_height), int(out_width), input_size[3]]

    elif function_['type'] == 'relu':
        return input_size

    elif function_['type'] == 'norm':
        return input_size

    elif function_['type'] == 'fc':
        return [input_size[0], function_['output_size']]

    elif function_['type'] == 'ph':
        return input_size


def chain_size_crunch(input_size, funcs):
    # This applies calc_size_after() to every one of the funcs
    updated = input_size
    for f in funcs:
        updated = calc_size_after(updated, f)
    return updated


# Helper function to help with fetching node data from the big dump
def fetch_node(nickname='no_nickname_given', graph=None, **kwargs):
    if graph is None:
        raise Exception('No node storage provided for fetch function!')
    if len(kwargs) > 0:
        dict_to_iterate = kwargs
    else:
        dict_to_iterate = {'nickname': nickname}

    matching = []
    for k, v in dict_to_iterate.items():
        contains_this_val = [graph.node[i] for i in
                             [ii for ii in graph.nodes()
                              if k in graph.node[ii]]
                             if graph.node[i][k] == v
                             and graph.node[i] not in matching]
        matching += [i for i in contains_this_val
                     if i not in matching]

    return matching


def initializer(kind='xavier', stddev=.01, seed=None):
    if kind == 'xavier':
        init = tf.contrib.layers.initializers.xavier_initializer(
            seed=seed if seed else randint(1, 1000))
    elif kind == 'trunc_norm':
        init = tf.truncated_normal_initializer(
            mean=0, stddev=stddev, seed=seed if seed else randint(1, 1000))
    else:
        raise ValueError('Please provide an appropriate initialization '
                         'method: xavier or trunc_norm')
    return init
