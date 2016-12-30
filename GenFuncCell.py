"""
The GenFuncCell object
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import RNNCell
from harbor import Harbor, Harbor_Dummy



class GenFuncCell(RNNCell):
    def __init__(self, 
                 harbor,
                 state_fs, 
                 out_fs, 
                 state_fs_kwargs,
                 out_fs_kwargs,
                 memory_kwargs,
                 output_size=None, 
                 state_size=None, 
                 seed=None, 
                 scope=None):
        function_lookup={'conv':tf.nn.conv2d,
                         'maxpool':tf.nn.max_pool,
                         'relu':tf.nn.relu,
                         'norm':tf.nn.local_response_normalization,
                         'fc':self.fc}
        """
        Input explanation:

        - harbor:
            The Harbor for this Node, the inputs are received here
        - state_fs:
            A list of functions to be called in order before memory update
        - out_fs:
            A list of functions to be called in order after memory update
        - state_fs_kwargs:
            Keyworded arguments to be passed into the state_fs above
        - out_fs_kwargs: 
            Keyworded arguments to be passed into the out_fs above
        - memory_kwargs: 
            Keyworded arguments to be used to determine how memory is worked
        - output_size: 
            Final output size of the Cell
        - state_size: 
            Size of the saved state in the memory
        - seed: 
            'Nuff said
        - scope: 
            Determines the scope in which the Cell is located

        """
        #=======
        # Check to see that state_fs is a list of functions, list-ify if not
        self._state_fs=state_fs if isinstance(state_fs,type([])) \
                                else [state_fs]
        # Check every item of self._state_fs to see if it is a function,
        # if not then convert from string to function using function_lookup {}
        self._state_fs=[function_lookup[i] if isinstance(i, basestring) \
                                           else i \
                        for i in self._state_fs]
        # Fetch all the keyword arguments for the pre-memory functions (list)
        self._state_fs_kwargs=state_fs_kwargs
        #=======
        # Check to see that out_fs is a list of functions, list-ify if not
        self._out_fs=out_fs if isinstance(out_fs,type([])) else [out_fs]
        # Check every item of self._out_fs to see if it is a function,
        # if not then convert from string to function using function_lookup {}
        self._out_fs=[function_lookup[i] if isinstance(i, basestring) \
                                           else i \
                        for i in self._out_fs]
        # Fetch all the keyword arguments for the post-memory functions (list)
        self._out_fs_kwargs=out_fs_kwargs
        #=======
        self._memory_kwargs=memory_kwargs
        #=======
        self._scope = type(self).__name__ if scope is None else scope
        self._output_size = output_size if isinstance(output_size,type([])) \
                                        else output_size.as_list()
        self._state_size = state_size if isinstance(state_size,type([])) \
                                      else state_size.as_list()

        # The zero_state functions are inherited from the RNNCell
        self.state = tf.zeros(self._state_size)
        self.state_old = tf.zeros(self._state_size)

        self.harbor=harbor

    # @property
    # def state_size(self):
    #     return self.state_size

    # @property
    # def output_size(self):
    #     return self.output_size

    # @property
    # def scope(self):
    #     return self._scope

    def __call__(self, input_): 
        # Input is a dict {'nickname':Tensor}
        prev=self.harbor(input_)

        # Each before-the-memory function, when run, will update the prev 
        # value and pass that to the next function
        for f in range(len(self._state_fs)):
            # The current function we're working with
            cur_f=self._state_fs[f]
            # The current function's args passed in (cur_f_args is a dict), 
            # everything here has been prepared outside of the Cell
            cur_f_args=self._state_fs_kwargs[f]
            # Plug in the input and open up the kwargs into the arguments of
            # the current function, and collect the output
            prev=cur_f(prev,**cur_f_args)

        # Now, we update the memory!
        self.state_old=self.state
        self.state=self.memory(in_layer=prev, **self._memory_kwargs)

        # Each after-the-memory function, when run, will update the 
        # self.output value and pass that to the next function
        for f in range(len(self._out_fs)):
            # The current function we're working with
            cur_f=self._out_fs[f]
            # The current function's args passed in (cur_f_args is a dict), 
            # everything here has been prepared outside of the Cell
            cur_f_args=self._out_fs_kwargs[f]
            # Plug in the input and open up the kwargs into the arguments of
            # the current function, and collect the output
            self.output=cur_f(self.output,**cur_f_args)

        return self.output, self.state



    def memory(self, in_layer=None, memory_decay=0, trainable=False):
        # Loop it's own OUTPUT into itself if no INPUT available, otherwise
        # loop the INPUT in, along with a scaled STATE. 
        # Return the resultant addition of scaled STATE and INPUT/OUTPUT
        if in_layer is None: in_layer = self.output
        name = tf.get_variable_scope().name
        initializer = tf.constant_initializer(value=memory_decay)
        mem = tf.get_variable(initializer=initializer,
                              shape=1,
                              trainable=trainable,
                              name='decay_param_%s'%(name))
        decay_factor = tf.sigmoid(mem)
        self.output = tf.mul(self.state, decay_factor) + in_layer
        return self.output

    def fc(input_, output_size):
        # Move everything into depth so we can perform a single matrix multiply.
        batch_size=input_.get_shape()[0].value
        name = tf.get_variable_scope().name
        reshape = tf.reshape(input_, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights_%s'%(self._scope), 
                        shape=[dim, output_size], 
                        initializer=tf.random_normal_initializer(0.5, 0.1))
        biases = _variable_on_cpu('biases_%s'%(scope), 
                                  [output_size], 
                                  tf.constant_initializer(0.1))
        mulss = tf.nn.relu(tf.matmul(reshape, weights) + biases)
        return mulss
