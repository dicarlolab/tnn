"""
Harbor Class Definition
"""
import tensorflow as tf


class Harbor(object):
    def __init__(self,
                 incoming_sizes,
                 node_name='',
                 policy=None,
                 **policy_kwargs):

        # Policy is a function that performs some set of actions on the input
        self.policy = policy if policy else Policy()
        self.shape_select = self.policy.shape_select
        self.combination_type = self.policy.combination_type
        self.name = node_name

        # Find the desired input size after Harbor processing
        self.desired_size = self.policy.reshape_size_to(incoming_sizes)

    def __call__(self, inputs):
        # inputs is a dictionary of {'nickname':Tensor}
        big_input_list = []
        print 'Inputs:', inputs
        for incoming_input in inputs:
            print 'Harbor call of cell %s for resizing node %s' % \
                (self.name, incoming_input)
            input_val = inputs[incoming_input]
            input_shape = input_val.get_shape().as_list()

            print 'TF Size:', input_shape

            # Function to create the TF node based on current input name,
            # shape, TF value and desired size - returns pool TF node that is
            # then added to big_input_list
            big_input_list.append(self.policy.tf_node_func(incoming_input,
                                                           input_shape,
                                                           input_val,
                                                           self.desired_size))

        # Now we're working outside of the individual input resizing loop
        # Let's combine all the inputs together:
        if self.combination_type == 'concat':
            out = tf.concat(3, big_input_list, name=self.name +
                            '_harbor_concat')
        elif self.combination_type == 'sum':
            out = tf.add_n(big_input_list, name=self.name + '_harbor_sum')

        print '  >> Harbor of %s - out size %s' % (self.name,
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
                 size_to_resize_to_func=None,
                 tf_node_func=None):
        self.shape_select = shape_select
        self.combination_type = combination_type
        self.size_to_resize_to_func = size_to_resize_to_func if \
            size_to_resize_to_func else self.reshape_size_for_long
        self.tf_node_func = tf_node_func if tf_node_func else self.tf_node_long

    def reshape_size_to(self, incoming_sizes):
        # incoming_sizes.items() =
        #               ( 'nickname' , ([size here],[[path],[path], ...]) )
        # Get the final shape from a pre-defined or input function:
        final_shape = self.size_to_resize_to_func(incoming_sizes)
        return final_shape

    def reshape_size_for_maxavg(self, incoming_sizes):
        # Find the max shape size in all the inputs:
        max_shape = max(incoming_sizes.items(), key=lambda x:
                        reduce(lambda p, q: p * q, x[1][0]))[1][0]
        return max_shape

    def reshape_size_for_long(self, incoming_sizes):
        # Find the max shape size in all the inputs:
        long_shape = max(incoming_sizes.items(),
                         key=lambda x: max(len(t) for t in x[1]))[1][0]
        return long_shape

    def reshape_size_for_short(self, incoming_sizes):
        # Find the max shape size in all the inputs:
        short_shape = min(incoming_sizes.items(),
                          key=lambda x: min(len(t) for t in x[1]))[1][0]
        return short_shape

    def reshape_size_for_min(self, incoming_sizes):
        # Find the max shape size in all the inputs:
        min_shape = min(incoming_sizes.items(),
                        key=lambda x: reduce(lambda p,
                                             q: p * q, x[1][0]))[1][0]
        return min_shape

    def tf_node_long(self,
                     incoming_input,
                     input_shape,
                     input_val,
                     desired_size,
                     name=''):

        def is_bigger(shape1, shape2):
            # Return True if shape1 has more neurons than shape2, False o/w
            if reduce(lambda x, y: x * y, shape1) > reduce(lambda x, y: x * y,
                                                           shape2):
                return True
            else:
                return False

        # If the number of dimensions in the input_shape is like an image:
        if len(input_shape) > 2:
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
                pool = tf.nn.max_pool(input_val,
                                      ksize=ksize,
                                      strides=strides,
                                      padding='VALID',
                                      name=name
                                      + '_' + incoming_input
                                      + '_harbor_maxpool')
            elif self.shape_select == 'avg':
                pool = tf.nn.avg_pool(input_val,
                                      ksize=ksize,
                                      strides=strides,
                                      padding='VALID',
                                      name=name
                                      + '_' + incoming_input
                                      + '_harbor_avgpool')
            elif self.shape_select == 'up':
                # Not sure if tf.image.resize_images is trainable...
                pool = tf.image.resize_images(input_val, desired_size[1:3])
            else:
                pool = input_val

            print '  >> Harbor of %s - resizing %s, want %s and got %s' % \
                (name,
                 incoming_input,
                 desired_size,
                 pool.get_shape().as_list())
            # Now append to a list of all the inputs
            return pool

        # If the number of dimensions is 2:
        else:
            return input_val
