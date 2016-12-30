"""
Harbor Class Definition
"""

from unicycle_settings import *
import tensorflow as tf

class Harbor(object):
    def __init__(self,desired_size,policy=HARBOR_MASTER_DEFAULT,node_name=''):
        self.desired_size=desired_size
        # Policy is a tuple of (pooling,combination), where pooling is the
        # type of pooling to be performed on the input and combination is the
        # type of combination function to use when combining all the inputs
        # together.
        self.policy=policy
        self.pooling_type=self.policy[0]
        self.combination_type=self.policy[1]
        self.name=node_name

    def __call__(self,inputs):
        # inputs is a dictionary of {'nickname':Tensor}
        big_input_list=[]
        for incoming_input in inputs:
            print('Harbor call for resizing node %s'%(incoming_input))
            input_=inputs[incoming_input]
            input_shape=input_.get_shape().as_list()
            
            print 'TF Size:', input_shape
            
            # Final size of pooling is f = (i-k)/s + 1, 
            # where i is the input size, k is the window size, and s is the
            # stride
            height_s=input_shape[1] // self.desired_size[1]
            width_s=input_shape[2] // self.desired_size[2]
            strides = [1,height_s,width_s,1]

            height_k=input_shape[1]-height_s*(self.desired_size[1]-1)
            width_k=input_shape[2]-width_s*(self.desired_size[2]-1)
            ksize=[1,height_k,width_k,1]

            # Do the resizing here
            if self.pooling_type=='max':
                pool = tf.nn.max_pool(input_,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          name=self.name+'_'+incoming_input+'_harbor_maxpool')
            elif self.pooling_type=='avg':
                pool = tf.nn.avg_pool(input_,
                          ksize=ksize,
                          strides=strides,
                          padding='VALID',
                          name=self.name+'_'+incoming_input+'_harbor_avgpool')
            elif self.pooling_type=='up':
                # Not sure if tf.image.resize_images is trainable...
                pool = tf.image.resize_images(input_, self.desired_size[1:3])

            # Now append to a list of all the inputs
            big_input_list.append(pool)

        # Now we're working outside of the individual input resizing loop
        # Let's combine all the inputs together:
        if self.combination_type=='concat':
            out = tf.concat(3, big_input_list,name=self.name+'_harbor_concat')
        elif self.combination_type=='sum':
            out = tf.add_n(big_input_list,name=self.name+'_harbor_sum')
        
        # Finally, return the resulting Tensor
        return out

class Harbor_Dummy(object):
    def __init__(self,node_name=''):
        self.name=node_name

    def __call__(self,inputs):
        inputs[0]