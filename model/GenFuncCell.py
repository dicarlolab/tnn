"""
The GenFuncCell object
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import RNNCell

verbose = True


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
        function_lookup = {'conv': tf.nn.conv2d,
                           'maxpool': tf.nn.max_pool,
                           'relu': tf.nn.relu,
                           'norm': tf.nn.local_response_normalization,
                           'fc': self.fc,
                           'placeholder': tf.placeholder}
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

        # ====== STATE_FUNCTIONS SETUP AND CLEANUP: ==========================

        # Check to see that state_fs is a list of functions, list-ify if not
        self._state_fs = state_fs if isinstance(state_fs, type([])) \
            else [state_fs]

        # Check every item of self._state_fs to see if it is a function,
        # if not then convert from string to function using function_lookup {}
        self._state_fs = [function_lookup[i] if isinstance(i, basestring)
                          else i for i in self._state_fs]

        # Fetch all the keyword arguments for the pre-memory functions (list)
        self._state_fs_kwargs = state_fs_kwargs

        # ====== OUT_FUNCTIONS SETUP AND CLEANUP: ============================

        # Check to see that out_fs is a list of functions, list-ify if not
        self._out_fs = out_fs if isinstance(out_fs, type([])) else [out_fs]
        # Check every item of self._out_fs to see if it is a function,
        # if not then convert from string to function using function_lookup {}
        self._out_fs = [function_lookup[i] if isinstance(i, basestring) else i
                        for i in self._out_fs]
        # Fetch all the keyword arguments for the post-memory functions (list)
        self._out_fs_kwargs = out_fs_kwargs

        # ====== MEMORY SETUP: ===============================================

        self._memory_kwargs = memory_kwargs

        # ====== STATES SETUP: ===============================================

        # States and outputs
        self._output_size = output_size if isinstance(output_size, type([])) \
            else output_size.as_list()
        self.outputs = [tf.zeros(self._output_size)]

        self._state_size = state_size if isinstance(state_size, type([])) \
            else state_size.as_list()
        self.states = [tf.zeros(self._state_size)]

        # ====== HARBOR AND MISC SETUP: ======================================

        # Harbor
        self.harbor = harbor

        # Scope of GenFuncCell and sub-functions
        self._scope = type(self).__name__ if scope is None else scope

    def __call__(self, input_, curstate):
        with tf.variable_scope(self._scope):
            print('GenFuncCell call of node %s' % (self._scope))

            # Check to see if input node:
            if len(input_) == 0:
                state = self.get_state()
                print()
                return state, state

            # Input is a dict {'nickname':Tensor}
            prev = self.harbor(input_)

            print('  Using Harbor output:', prev)
            print('  GenFuncCell of node %s >>  post-Harbor size %s' % (
                self._scope,
                prev.get_shape().as_list())
            )

            # Each before-the-memory function, when run, will update the prev
            # value and pass that to the next function
            for f in range(len(self._state_fs)):
                # The current function we're working with
                cur_f = self._state_fs[f]
                # The current function's args passed in (cur_f_args is a dict),
                # everything here has been prepared outside of the Cell
                cur_f_args = self._state_fs_kwargs[f]
                # Plug in the input and open up the kwargs into the arguments
                # of the current function, and collect the output
                prev = cur_f(prev, **cur_f_args)

            print('  GenFuncCell of node %s >>  post-state size %s' % (
                self._scope,
                prev.get_shape().as_list())
            )

            print('  GenFuncCell of node %s >>  pre-memory state size %s' % (
                self._scope,
                self.get_state().get_shape().as_list())
            )
            # Now, we update the memory!
            new_state = self.memory(state=curstate,
                                    in_layer=prev,
                                    **self._memory_kwargs)

            print('  GenFuncCell of node %s >>  post-memory state size %s' % (
                self._scope,
                new_state.get_shape().as_list())
            )

            new_output = new_state
            # Each after-the-memory function, when run, will update the
            # self.output value and pass that to the next function
            for f in range(len(self._out_fs)):
                # The current function we're working with
                cur_f = self._out_fs[f]
                # The current function's args passed in (cur_f_args is a dict),
                # everything here has been prepared outside of the Cell
                cur_f_args = self._out_fs_kwargs[f]
                # Plug in the input and open up the kwargs into the arguments
                # of the current function, and collect the output
                new_output = cur_f(new_output, **cur_f_args)

            print('  GenFuncCell of node %s >>  post-out-func size %s\n' % (
                self._scope,
                new_output.get_shape().as_list())
            )

            return new_output, new_state

    def get_state(self, t=0):
        # Return the topmost state or -t_th state (0 is current, 1 is previous)
        if t > len(self.states):
            raise Exception('GenFuncCell trying to access nonexistent state')
        return self.states[-t]

    def get_output(self, t=0):
        # Return the topmost state or -t_th state (0 is current, 1 is previous)
        if t > len(self.outputs):
            # raise Exception('GenFuncCell trying to access nonexistent output')
            return self.zero_state()
        return self.outputs[-t]

    def update_states(self, new):
        self.states.append(new)

    def update_outputs(self, new):
        self.outputs.append(new)

    def memory(self, state, in_layer=None, memory_decay=0, trainable=False):
        # Loop it's own OUTPUT into itself if no INPUT available, otherwise
        # loop the INPUT in, along with a scaled STATE.
        # Return the resultant addition of scaled STATE and INPUT/OUTPUT
        if in_layer is None:
            in_layer = self.get_output()
        initializer = tf.constant_initializer(value=memory_decay)
        mem = tf.get_variable(initializer=initializer,
                              shape=1,
                              trainable=trainable,
                              name='decay_param_%s' % (self._scope))
        print('    %s MEMORY CALLED! Decay param name: %s' % (
            self._scope, mem.name))
        decay_factor = tf.sigmoid(mem)
        new = tf.mul(state, decay_factor) + in_layer
        return new

    def fc(self, input_, output_size):
        # Move everything into depth so we can perform a single matrix mult.
        batch_size = input_.get_shape()[0].value
        reshape = tf.reshape(input_, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable(
            'weights_%s' % (self._scope),
            shape=[dim, output_size],
            initializer=tf.random_normal_initializer(0.5, 0.1))
        biases = tf.get_variable(
            'biases_%s' % (self._scope),
            shape=[output_size],
            initializer=tf.constant_initializer(0.1))
        mulss = tf.nn.relu(tf.matmul(reshape, weights) + biases)
        return mulss

    def zero_state(self):
        return tf.zeros(self._state_size)
