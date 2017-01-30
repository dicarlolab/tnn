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
        self.policy = policy if policy else Policy(**policy_kwargs)

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
            # If image:
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
                 tf_node_func=None):
        # The characteristic of the shape to be selected and all input to
        # be resized to:
        #   long    -> longest path,
        #   short   -> shortest path,
        #   max     -> maximum size,
        #   min     -> minimum size
        #   avg     -> average all the sizes (this is not yet implemented)
        self.shape_select = shape_select

        # How to combine the inputs (for Harbor use):
        #   concat  -> concatenate all inputs in depth
        #   sum     -> sum all the inputs (must be same size!)
        self.combination_type = combination_type

        # The final size to resize to based on shape_select from above:
        self.tensor_calculate_func = tensor_calculate_func if \
            tensor_calculate_func else self.concat_choice_final_tensor

    # def final_output_size_func(self, incoming_sizes):
    #     # incoming_sizes.items() =
    #     #               ( 'nickname' , ([size here],[[path],[path], ...]) )
    #     # Get the final shape from a pre-defined or input function. Note that
    #     # this is a bit redundant and the function 'size_to_resize_to_func'
    #     # essentially be called directly without this wrapper in place. The
    #     # only reason this is here is for modularity and simplicity (we
    #     # define what the transformation function should be when the Policy
    #     # is initialized, and then that transformation function is never
    #     # directly touched again, we only deal with the higher-level
    #     # 'final_output_size' function)
    #     final_shape = self.size_to_resize_to_func(incoming_sizes)
    #     return final_shape

    def max_choice_final_tensor(self, incoming_sizes):
        # Find the max shape size in all the inputs:
        max_shape = max(incoming_sizes.items(), key=lambda x:
                        reduce(lambda p, q: p * q, x[1][0]))[1][0]
        return max_shape

    def avg_choice_final_tensor(self, incoming_sizes):
        # Find the average shape size in all the inputs:
        raise NotImplementedError()

    def long_choice_final_tensor(self,
                                 input_nickname,
                                 input_tensor,
                                 desired_size,
                                 name=''):

        # If the number of dimensions in the input_shape is like an image:
        if len(input_shape) == 4:
            # Final size of pooling is f = (i-k)/s + 1,
            # where i is the input size, k is the window size, and s is the
            # stride
            height_s = input_shape[1] // desired_size[1]
            width_s = input_shape[2] // desired_size[2]
            strides = [1, height_s, width_s, 1]

            height_k = input_shape[1] - height_s * (desired_size[1] - 1)
            width_k = input_shape[2] - width_s * (desired_size[2] - 1)
            ksize = [1, height_k, width_k, 1]

            # Do the resizing here
            if self.shape_select == 'max':
                pool = tf.nn.max_pool(input_tensor,
                                      ksize=ksize,
                                      strides=strides,
                                      padding='VALID',
                                      name=name
                                      + '_' + input_nickname
                                      + '_harbor_maxpool')
            elif self.shape_select == 'avg':
                pool = tf.nn.avg_pool(input_tensor,
                                      ksize=ksize,
                                      strides=strides,
                                      padding='VALID',
                                      name=name
                                      + '_' + input_nickname
                                      + '_harbor_avgpool')
            elif self.shape_select == 'up':
                # Not sure if tf.image.resize_images is trainable...
                pool = tf.image.resize_images(input_tensor, desired_size[1:3])
            else:
                pool = input_tensor

            print '  >> Harbor Policy of %s - resizing %s, want %s and got %s' % \
                (name,
                 input_nickname,
                 desired_size,
                 pool.get_shape().as_list())
            # Now append to a list of all the inputs
            return pool

        # If the number of dimensions is 2:
        else:
            return input_tensor

        # Find the shape size of the input with the longest path:
        long_shape = max(incoming_sizes.items(),
                         key=lambda x: max(len(t) for t in x[1]))[1][0]
        return long_shape

    def short_choice_final_tensor(self, incoming_sizes):
        # Find the shape size of the input with the shortest path:
        short_shape = min(incoming_sizes.items(),
                          key=lambda x: min(len(t) for t in x[1]))[1][0]
        return short_shape

    def min_choice_final_tensor(self, incoming_sizes):
        # Find the min shape size in all the inputs:
        min_shape = min(incoming_sizes.items(),
                        key=lambda x: reduce(lambda p,
                                             q: p * q, x[1][0]))[1][0]
        return min_shape

    def concat_choice_final_tensor(self,
                                   incoming_sizes=None,
                                   return_desired_size=False,
                                   input_tensor=None,
                                   **kwargs):
        if return_desired_size:
            if not incoming_sizes:
                raise Exception('Need to supplement incoming_sizes if return_desired_size boolean is set to True!')
            # Find the shape of all the inputs concatenated:
            batch_size = incoming_sizes.items()[0][1][0][0]
            rolling_sum = 0
            for n, i in incoming_sizes.items():
                rolling_sum += reduce(lambda x, y: x * y, i[0][1:])
                print(rolling_sum)
            concat_shape = [batch_size, rolling_sum]
            return concat_shape
        else:
            # otherwise, we return the same tensor to be concatenated:
            input_shape = input_tensor.get_shape().as_list()
            if len(input_shape) < 4:
                return tf.reshape(input_tensor, [input_shape[0], -1])
            else:
                return input_tensor
