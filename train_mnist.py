"""
Unicycle Training using TF-Utils
"""

from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
import tensorflow as tf

from tfutils import optimizer   # , base #, data
import mod_data as data
import mod_base as base

# little hack to include model folder
bp = os.path.dirname(os.path.realpath('.')).split(os.sep)
modpath = os.sep.join(bp + ['tconvnet/model'])
sys.path.insert(0, modpath)
import unicycle


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# host = os.uname()[1]
# if host.startswith('node') or host == 'openmind7':  # OpenMind
#     DATA_PATH = '/om/user/qbilius/imagenet/data.raw'
# else:  # agents
DATA_PATH = 'MNIST_data/'
RESTORE_VAR_FILE = 'computed/alexnet_test/'


def in_top_k(inputs, outputs, target):
    return {'top1': tf.nn.in_top_k(outputs, inputs[target], 1),
            'top5': tf.nn.in_top_k(outputs, inputs[target], 5)}


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res


def exponential_decay(global_step,
                      learning_rate=.01,
                      decay_factor=.95,
                      decay_steps=1,
                      ):
    # Decay the learning rate exponentially based on the number of steps.
    if decay_factor is None:
        lr = learning_rate  # just a constant.
    else:
        # Calculate the learning rate schedule.
        lr = tf.train.exponential_decay(
            learning_rate,  # Base learning rate.
            global_step,  # Current index into the dataset.
            decay_steps,  # Decay step
            decay_factor,  # Decay rate.
            staircase=True)
    return lr


N_TRAIN = 55000
N_VAL = 5000
N_TRAIN_VAL = 5000

BATCH_SIZE = 256
NUM_BATCHES_PER_EPOCH = N_TRAIN // BATCH_SIZE
IMAGE_SIZE_CROP = 28    # 28 for MNIST, 224 for ImageNet

params = {
    'save_params': {
        'host': 'localhost',
        'port': 32108,
        'dbname': 'tconvnet-mnist-test',
        'collname': 'mnist',
        'exp_id': 'trainval0',

        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 5,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 300,
        'save_filters_freq': 3000,
        'cache_filters_freq': 3000,
        # 'cache_dir': None,  # defaults to '~/.tfutils'
    },

    'load_params': {
        # 'host': 'localhost',
        # 'port': 31001,
        # 'dbname': 'alexnet-test',
        # 'collname': 'alexnet',
        # 'exp_id': 'trainval0',
        'do_restore': False,
        'load_query': None
    },

    'model_params': {
        'func': unicycle.mnist_tfutils,
        'seed': 0,
        'norm': False  # do you want local response normalization?
    },

    'train_params': {
        'data_params': {
            'func': data.MNIST,
            'data_path': DATA_PATH,
            'group': 'train',
            'batch_size': 1
        },
        'queue_params': {
            'queue_type': 'fifo',
            'batch_size': BATCH_SIZE,
            'n_threads': 4,
            'seed': 0,
        },
        'thres_loss': 100000,
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH  # number of steps to train
    },

    'loss_params': {
        'targets': 'labels',
        'agg_func': tf.reduce_mean,
        'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': .01,
        'decay_rate': .95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.MomentumOptimizer,
        'clip': True,
        'momentum': .9
    },

    'validation_params': {
        'topn': {
            'data_params': {
                'func': data.MNIST,
                'data_path': DATA_PATH,  # path to image database
                'group': 'validation'
            },
            'targets': {
                'func': in_top_k,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE,
                'n_threads': 4,
                'seed': 0,
            },
            'num_steps': N_VAL // BATCH_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        },
    },

    'log_device_placement': False,  # if variable placement has to be logged
    'allow_memory_growth': True,
}


def main(custom_step_limit=None):
    base.get_params()
    if custom_step_limit and isinstance(custom_step_limit, type(123)):
        params['train_params']['num_steps'] = custom_step_limit
    base.train_from_params(**params)
    return True


if __name__ == '__main__':
    main()
