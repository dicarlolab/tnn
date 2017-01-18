"""
AlexNet template function
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from ConvRNNCell import ConvRNNCell

def alexnet(input_spatial_size=224,
            batch_size=256,
            init_weights='xavier',
            weight_decay=.0005,
            memory_decay=None,
            memory_trainable=False,
            dropout=.5,
            train=True,
            seed=None
            ):
    dropout = dropout if train else None

    class AlexNetCell(ConvRNNCell):

        def __init__(self, seed=None, *args, **kwargs):
            super(AlexNetCell, self).__init__(seed=seed, *args, **kwargs)

        def conv(self,
                 activation='relu',
                 init=init_weights,
                 weight_decay=weight_decay,
                 memory_decay=memory_decay,
                 memory_trainable=memory_trainable,
                 *args, **kwargs):
            super(AlexNetCell, self).conv(activation=activation,
                                          init=init_weights,
                                          weight_decay=weight_decay,
                                          memory_decay=memory_decay,
                                          memory_trainable=memory_trainable,
                                          *args, **kwargs)

        def fc(self,
               activation='relu',
               dropout=dropout,
               init=init_weights,
               weight_decay=weight_decay,
               memory_decay=memory_decay,
               memory_trainable=memory_trainable,
               *args, **kwargs):
            super(AlexNetCell, self).conv(activation=activation,
                                          dropout=dropout,
                                          init=init,
                                          weight_decay=weight_decay,
                                          memory_decay=memory_decay,
                                          memory_trainable=memory_trainable,
                                          *args, **kwargs)


    class Conv1(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 8 - 1,
                           input_spatial_size // 8 - 1, 64]
            state_size = [batch_size, input_spatial_size // 4,
                          input_spatial_size // 4, 64]
            super(Conv1, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv1'):
                self.conv(inputs, state, ksize=11, stride=4, stddev=.01, bias=0)
                self.norm(depth_radius=4, bias=1, alpha=.001 / 9.0, beta=.75)
                self.pool(3, 2, padding='VALID')
                return self.output, self.state

    class Conv2(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 16 - 1,
                           input_spatial_size // 16 - 1, 192]
            state_size = [batch_size, input_spatial_size // 8 - 1,
                          input_spatial_size // 8 - 1, 192]
            super(Conv2, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv2'):
                self.conv(inputs, state, ksize=5, stride=1, stddev=.01, bias=1)
                self.norm(depth_radius=4, bias=1, alpha=.001 / 9.0, beta=.75)
                self.pool(3, 2, padding='VALID')
                return self.output, self.state

    class Conv3(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 16 - 1,
                           input_spatial_size // 16 - 1, 384]
            state_size = [batch_size, input_spatial_size // 16 - 1,
                          input_spatial_size // 16 - 1, 384]
            super(Conv3, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv3'):
                self.conv(inputs, state, ksize=3, stride=1, stddev=.01, bias=0)
                return self.output, self.state

    class Conv4(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 16 - 1,
                           input_spatial_size // 16 - 1, 384]
            state_size = [batch_size, input_spatial_size // 16 - 1,
                          input_spatial_size // 16 - 1, 384]
            super(Conv4, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv4'):
                self.conv(inputs, state, ksize=3, stride=1, stddev=.01, bias=1)
                return self.output, self.state

    class Conv5(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 32,
                           input_spatial_size // 32, 256]
            state_size = [batch_size, input_spatial_size // 16 - 1,
                          input_spatial_size // 16 - 1, 256]
            super(Conv5, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv5'):
                self.conv(inputs, state, ksize=3, stride=1, stddev=.01, bias=1)
                self.pool(3, 2, padding='VALID')
                return self.output, self.state

    class FC6(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, 4096]
            state_size = [batch_size, 4096]
            super(FC6, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('FC6'):
                self.fc(inputs, state, stddev=.01, bias=1)
                return self.output, self.state

    class FC7(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, 4096]
            state_size = [batch_size, 4096]
            super(FC7, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('FC7'):
                self.fc(inputs, state, stddev=.01, bias=1)
                return self.output, self.state

    class FC8(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, 1000]
            state_size = [batch_size, 1000]
            super(FC8, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('FC8'):
                self.fc(inputs, state, activation=None, dropout=None, stddev=.01, bias=0)
                return self.output, self.state

    layers = [Conv1, Conv2, Conv3, Conv4, Conv5, FC6, FC7, FC8]

    return layers