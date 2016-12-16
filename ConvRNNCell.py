"""
The ConvRNNCell object
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import RNNCell
from tfutils.model import ConvNet


class ConvRNNCell(ConvNet, RNNCell):

    def __init__(self, output_size, state_size, seed=None, scope=None):
        super(ConvRNNCell, self).__init__(seed=seed)
        self.scope = type(self).__name__ if scope is None else scope
        self._output_size = output_size
        self._state_size = state_size
        self.state = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state=None):
        raise NotImplementedError

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor

        Args:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.
        Returns:
            A tensor of shape `state_size` filled with zeros.
        """
        zeros = tf.zeros(self.state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def memory(self, state, memory_decay=0, trainable=False, in_layer=None):
        if in_layer is None: in_layer = self.output
        initializer = tf.constant_initializer(value=memory_decay)
        mem = tf.get_variable(initializer=initializer,
                              shape=1,
                              trainable=trainable,
                              name='decay_param')
        decay_factor = tf.sigmoid(mem)
        self.output = tf.mul(state, decay_factor) + in_layer
        return self.output

    def conv(self, inputs, state,
             activation='relu',
             memory_decay=0,
             memory_trainable=False,
             *args, **kwargs):
        # Perform the convolution
        super(ConvRNNCell, self).conv(in_layer=inputs,
                                      out_shape=self.state_size[-1],
                                      activation=None,
                                      *args, **kwargs)
        # Update the state
        self.state = self.memory(state,
                                 memory_decay=memory_decay,
                                 trainable=memory_trainable)
        # 
        if activation is not None:
            self.activation(kind=activation)
        name = tf.get_variable_scope().name
        self.params[name]['conv']['activation'] = activation
        self.params[name]['conv']['memory_decay'] = memory_decay
        self.params[name]['conv']['memory_trainable'] = memory_trainable
        return self.output

    def fc(self, inputs, state,
           activation='relu',
           dropout=None,
           memory_decay=0,
           memory_trainable=False,
           *args, **kwargs):
        super(ConvRNNCell, self).fc(in_layer=inputs,
                                    out_shape=self.state_size[-1],
                                    activation=None,
                                    dropout=None,
                                    *args, **kwargs)
        self.state = self.memory(state,
                                 memory_decay=memory_decay,
                                 trainable=memory_trainable)
        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            self.activation(dropout=dropout)
        name = tf.get_variable_scope().name
        self.params[name]['fc']['activation'] = activation
        self.params[name]['fc']['dropout'] = dropout
        self.params[name]['fc']['memory_decay'] = memory_decay
        self.params[name]['fc']['memory_trainable'] = memory_trainable
        return self.output