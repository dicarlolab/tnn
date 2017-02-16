"""
Harbor Class Definition
"""
import tensorflow as tf
import numpy as np


class Harbor(object):
    def __init__(self,
                 incoming_sizes,
                 node_name='',
                 policy=None,
                 **policy_kwargs):
        """
        Harbor Class for accepting inputs into the GenFuncCell and resizing
        them before the in-cell functions are executed

        Inputs:

        - incoming_sizes:
            The incoming sizes and paths of the inputs.
            {'nickname': (size, all_simple_paths), ...}
        - node_name:
            Name of the GenFuncCell node that the Harbor belongs to
        - policy (optional, default=None):
            The Policy object with the parameter definition for this Harbor
        - policy_kwargs (optional):
            Keyworded arguments to be passed into a new Policy if none is
            provided above
        """

        # ====== EITHER USE, OR CREATE AND USE POLICY: =======================

        # Policy is a function-carrying object that performs some set of
        # actions on the input
        self.policy = policy if policy else Policy(name=node_name,
                                                   **policy_kwargs)

        # ====== GET SETUP PARAMS FROM POLICY: ===============================

        # This indicates which characteristic of the input to consider in
        # picking the proper output size (size, length of path, etc)
        self.shape_select = self.policy.shape_select

        # How to combine all the inputs at the end
        self.combination_type = self.policy.combination_type

        # Find the desired input size after Harbor processing
        self.desired_size = \
            self.policy.tensor_calculate_func(incoming_sizes,
                                              return_desired_size=True)

        # ====== MISC: =======================================================

        # Name of current cell (for scoping and logging mostly)
        self.name = node_name

    def __call__(self, inputs):
        """
        The __call__ function of the Harbor actually
            1. Resizes all the incoming Tensors
            2. Combines them
            3. Returns the resulting Tensor

        Inputs:

        - inputs:
            The incoming inputs into the Harbor (or technically into the
            GenFuncCell that the Harbor belongs to)
            {'nickname':Tensor, ...}

        Returns:
        - TF graph node if the Harbor has inputs
        """

        # Check to see if the Harbor belongs to an input cell
        if len(inputs) == 0:
            raise NotImplementedError

        big_input_list = []
        print ' Harbor call of cell %s:' % (self.name)
        print ' Inputs:', inputs
        for (nickname, input_tensor) in inputs.items():
            print ' Resizing output of node %s' % (nickname)

            print ' TF Size:', input_tensor.get_shape().as_list()

            # Function to create the TF node based on current input name,
            # shape, TF value and desired size - returns pool TF node that is
            # then added to big_input_list
            #
            #
            #   ONLY WORKS WITH CONCAT FOR NOW!!!
            #
            big_input_list.append(
                self.policy.tensor_calculate_func(
                    input_nickname=nickname,
                    input_tensor=input_tensor,
                    desired_size=self.desired_size))

        # Now we're working outside of the individual input resizing loop
        # Let's combine all the inputs together:
        if self.combination_type == 'concat':
            # If first input is an image:
            print ' Harbor post-policy inputs:'
            for i in big_input_list:
                print ' +- ', i
            if len(big_input_list[0].get_shape().as_list()) == 4:
                out = tf.concat(3, big_input_list, name=self.name +
                                '_harbor_concat')
            # If not image:
            else:
                out = tf.concat(1, big_input_list, name=self.name +
                                '_harbor_concat')
        elif self.combination_type == 'sum':
            out = tf.add_n(big_input_list, name=self.name + '_harbor_sum')

        print ' Harbor of %s >>  out size %s' % (self.name,
                                                 out.get_shape().as_list())

        # Finally, return the resulting Tensor
        return out

    def get_desired_size(self):
        # Simple function to return the pre-computed desired size
        return self.desired_size


class Harbor_Dummy(object):
    def __init__(self,
                 desired_size,
                 node_name='',
                 input_=False):
        self.name = node_name
        self.desired_size = desired_size
        self.input_ = input_

    def __call__(self, inputs):
        return inputs


class Policy(object):
    def __init__(self,
                 shape_select='long',
                 combination_type='concat',
                 tensor_calculate_func=None,
                 tf_node_func=None,
                 name=''):
        # The characteristic of the shape to be selected and all input to
        # be resized to:
        #   long    -> longest path,
        #   short   -> shortest path,
        #   max     -> maximum size,
        #   min     -> minimum size
        #   avg     -> average all the sizes (this is not yet implemented)
        #   concat  -> concatenate all the tensors into a large one
        self.shape_select = shape_select

        # How to combine the inputs (for Harbor use):
        #   concat  -> concatenate all inputs in depth
        #   sum     -> sum all the inputs (must be same size!)
        self.combination_type = combination_type

        # The final size to resize to based on shape_select from above:
        self.tensor_calculate_func = tensor_calculate_func if \
            tensor_calculate_func else self.choice_final_tensor

        self.name = name

    def choice_final_tensor(self,
                            incoming_sizes=None,
                            input_nickname=None,
                            input_tensor=None,
                            desired_size=None,
                            return_desired_size=False,
                            **kwargs):
        """
        Helper function for either
            a) determining the proper size of the final Harbor output
            b) returning the resized input based on the desired output size

        Inputs for case a):

        - return_desired_size=True
            Setting return_desired_size to True puts the function into size
            calculation mode, and the incoming_sizes argument is then used to
            find the proper output size
        - incoming_sizes:
            The incoming sizes and paths of the inputs.
            {'nickname': (size, all_simple_paths), ...}

        Inputs for case b):

        - return_desired_size=False
            Setting return_desired_size to False makes the function work with
            individual inputs using input_nickname, input_tensor and
            desired_size arguments to come up with the final tensor.
        - input_nickname
            The nickname of the current input
        - input_tensor
            The current input's TF tensor
        - desired_size
            The desired size to be resized to
        """

        # By default this is the long_choice_final_tensor function:
        return self.long_choice_final_tensor(incoming_sizes,
                                             input_nickname,
                                             input_tensor,
                                             desired_size,
                                             return_desired_size,
                                             **kwargs)

    def long_choice_final_tensor(self,
                                 incoming_sizes=None,
                                 input_nickname=None,
                                 input_tensor=None,
                                 desired_size=None,
                                 return_desired_size=False,
                                 **kwargs):
        """
        Function that selects the input with the longest path as the one to
        compare the desired input size to.

        For more general description of helper functions see comment in the
        choice_final_tensor function
        """

        if return_desired_size:
            # Find the shape size of the input with the longest path:
            long_shape = max(incoming_sizes.items(),
                             key=lambda x: max(len(t) for t in x[1]))[1][0]
            return long_shape
        else:
            # Get the input shape
            input_shape = input_tensor.get_shape().as_list()

            # If the number of dimensions in the input_shape is like an image:
            if len(input_shape) > 2:

                # If the number of dimensions in the input_shape is like an image:
                if len(input_shape) == 3:
                    # Reshape to 4:
                    input_tensor = tf.reshape(input_tensor, [input_shape[0],
                                                             (input_shape[2])**0.5,
                                                             (input_shape[2])**0.5,
                                                             input_shape[1]],
                                              name='reshape_to_4')

                # If the shape of the current input is the desired shape, pass:
                if input_shape == desired_size:
                    return input_tensor

                # If the shape of the current input is more than the desired
                # size, we pool down
                if reduce(lambda x,
                          y: x * y,
                          input_shape) > reduce(lambda x,
                                                y: x * y,
                                                desired_size):
                    # Final size of pooling is f = (i-k)/s + 1,
                    # where i is the input size, k is the filter size, and s
                    # is the stride
                    h_s = input_shape[1] // desired_size[1]
                    w_s = input_shape[2] // desired_size[2]
                    strides = [1, h_s, w_s, 1]

                    h_k = input_shape[1] - h_s * (desired_size[1] - 1)
                    w_k = input_shape[2] - w_s * (desired_size[2] - 1)
                    ksize = [1, h_k, w_k, 1]

                    # maxpool
                    pool = tf.nn.max_pool(input_tensor,
                                          ksize=ksize,
                                          strides=strides,
                                          padding='VALID',
                                          name=self.name
                                          + '_' + input_nickname
                                          + '_harbor_maxpool')

                    print '  >> Harbor Policy of %s - resizing %s, want %s and got %s' % \
                        (self.name,
                         input_nickname,
                         desired_size,
                         pool.get_shape().as_list())
                    # Now append to a list of all the inputs
                    return pool

                # Otherwise, we upscale
                else:
                    up_mult = desired_size[1] // input_shape[1]
                    # Unpool
                    dim = len(input_shape[1:-1])
                    out = (tf.reshape(input_tensor, [-1] + input_shape[-dim:]))
                    for i in range(dim, 0, -1):
                        out = tf.concat(i, [out, tf.zeros_like(out)])
                    out_size = [-1] \
                        + [s * up_mult for s in input_shape[1:-1]] \
                        + [input_shape[-1]]
                    out = tf.reshape(out, out_size)
                    return out

            # If the number of dimensions is 2:
            else:
                return input_tensor

    def short_choice_final_tensor(self, incoming_sizes):
        # Find the shape size of the input with the shortest path:
        short_shape = min(incoming_sizes.items(),
                          key=lambda x: min(len(t) for t in x[1]))[1][0]
        return short_shape

    def max_choice_final_tensor(self, incoming_sizes):
        # Find the max shape size in all the inputs:
        max_shape = max(incoming_sizes.items(), key=lambda x:
                        reduce(lambda p, q: p * q, x[1][0]))[1][0]
        return max_shape

    def min_choice_final_tensor(self, incoming_sizes):
        # Find the min shape size in all the inputs:
        min_shape = min(incoming_sizes.items(),
                        key=lambda x: reduce(lambda p,
                                             q: p * q, x[1][0]))[1][0]
        return min_shape

    def avg_choice_final_tensor(self, incoming_sizes):
        # Find the average shape size in all the inputs:
        raise NotImplementedError()

    def concat_choice_final_tensor(self,
                                   incoming_sizes=None,
                                   input_nickname=None,
                                   input_tensor=None,
                                   desired_size=None,
                                   return_desired_size=False,
                                   **kwargs):
        """
        Concatenation function (we take all the inputs and concatenate them
        along the size dimension for FC and along channel dimension for CONV).

        For more general description of helper functions see comment in the
        choice_final_tensor function
        """
        if return_desired_size:
            # If we're just trying to find the final Harbor output size,
            if not incoming_sizes:
                raise Exception('Need to supplement incoming_sizes if \
                    return_desired_size boolean is set to True!')

            # Find the shape of all the inputs concatenated. Remember that the
            # data stored in incoming_sizes dictionary is:
            #   {'nickname': (size, all_simple_paths), ...}
            # so incoming_sizes.items() is a list of tuples:
            #   [('nickname',(size, all_simple_paths)), (...)]
            #       1st index points to the input
            #       2nd index chooses the nickname (0) or the info tuple (1)
            #       3rd index selects size (0) or the path list (1)
            #       4th index selects a particular dimension or a path
            batch_size = incoming_sizes.items()[0][1][0][0]
            rolling_sum = 0
            for n, i in incoming_sizes.items():
                # Multiply all the non-batch dimensions together and add for
                # each one of the inputs
                rolling_sum += reduce(lambda x, y: x * y, i[0][1:])
            concat_shape = [batch_size, rolling_sum]
            return concat_shape

        else:
            # If we need the final resized tensors, we reshape the input to
            # be [batch, -1] sized if the number of dimensions is less than 4
            # (i.e. if the input is not a convolutional layer)
            input_shape = input_tensor.get_shape().as_list()
            batch_size = input_shape[0]
            if len(input_shape) < 4:
                return tf.reshape(input_tensor, [batch_size, -1])
            else:
                return input_tensor
