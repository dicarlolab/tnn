"""
Unicycle Utility Functions
"""

from math import floor, ceil
import tensorflow as tf

def create_kwargs_conv(fn, input_size, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict={}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type']=='conv', 'Type has to be CONV, but is %s'%(fn['type'])
    filter_tensor=tf.get_variable('filter_tensor_%s'%(name),
                                  fn['filter_size']\
                                  +[input_size[-1]]\
                                  +[fn['num_filters']],
                                initializer=tf.random_uniform_initializer(0,1)
                                 )
    out_dict['filter']=filter_tensor
    out_dict['strides']=fn['stride']
    out_dict['padding']=fn['padding']
    out_dict['name']=str(name) if isinstance(name, basestring) \
                               else 'conv_%s'%(str(randint(1,1000)))

    return out_dict, calc_size_after(input_size,fn)

def create_kwargs_maxpool(fn, input_size, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict={}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type']=='maxpool', 'Type has to be MAXPOOL, but is %s'%(fn['type'])
    
    out_dict['ksize']=fn['k_size']
    out_dict['strides']=fn['stride']
    out_dict['padding']=fn['padding']
    out_dict['name']=str(name) if isinstance(name, basestring) \
                               else 'maxpool_%s'%(str(randint(1,1000)))
    return out_dict, calc_size_after(input_size,fn)

def create_kwargs_relu(fn, input_size, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict={}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type']=='relu', 'Type has to be RELU, but is %s'%(fn['type'])

    out_dict['name']=str(name) if isinstance(name, basestring) \
                               else 'relu_%s'%(str(randint(1,1000)))
    return out_dict, calc_size_after(input_size,fn)

def create_kwargs_norm(fn, input_size, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict={}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type']=='norm', 'Type has to be NORM, but is %s'%(fn['type'])
    
    out_dict['depth_radius']=fn['depth_radius']
    out_dict['bias']=fn['bias']
    out_dict['alpha']=fn['alpha']
    out_dict['name']=str(name) if isinstance(name, basestring) \
                               else 'norm_%s'%(str(randint(1,1000)))
    return out_dict, calc_size_after(input_size,fn)

def create_kwargs_fc(fn, input_size):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict={}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type']=='fc', 'Type has to be FC, but is %s'%(fn['type'])
    
    out_dict['output_size']=fn['output_size']
    return out_dict, calc_size_after(input_size,fn)

def assemble_function_kwargs(functions,input_size,nickname):
    to_be_passed_in_state_kwargs=[]
    cur_size=input_size[:]
    for f in functions:
        f_type=f['type']
        temp_kwarg={}
        if f_type=='conv':
            temp_kwarg,cur_size=create_kwargs_conv(f, cur_size, nickname)
        elif f_type=='maxpool':
            temp_kwarg,cur_size=create_kwargs_maxpool(f, cur_size, nickname)
        elif f_type=='relu':
            temp_kwarg,cur_size=create_kwargs_relu(f, cur_size, nickname)
        elif f_type=='norm':
            temp_kwarg,cur_size=create_kwargs_norm(f, cur_size, nickname)
        elif f_type=='fc':
            temp_kwarg,cur_size=create_kwargs_fc(f, cur_size)
        to_be_passed_in_state_kwargs.append(temp_kwarg)
    return to_be_passed_in_state_kwargs






def reshape_size_to(incoming_sizes,current_policy):
    # Find the max and min shape sizes in all the inputs:
    # incoming_sizes.items() = ( 'nickname' , ([size here],1) )
    max_shape=max(incoming_sizes.items(), key=lambda x: x[1][0][1])[1][0]
    min_shape=min(incoming_sizes.items(), key=lambda x: x[1][0][1])[1][0]

    return min_shape if current_policy[0] in ['max','avg'] else max_shape

def calc_size_after(input_size,function_):
    if function_['type']=='conv':
        if function_['padding']=='valid':
            out_height = int(ceil(float(input_size[1]) \
                                  / float(function_['stride'])))
            out_width  = int(ceil(float(input_size[2]) \
                                  / float(function_['stride'])))
        elif function_['padding']=='same':
            out_height = int(ceil(float(input_size[1] \
                                        -function_['filter_size'][0]+1) \
                                  / float(function_['stride'])))
            out_width  = int(ceil(float(input_size[2] \
                                        -function_['filter_size'][1]+1) \
                                  / float(function_['stride'])))
        return [input_size[0],out_height,out_width,function_['num_filters']]

    elif function_['type']=='maxpool':
        if function_['padding']=='valid':
            out_height = floor(float(input_size[1]-function_['k_size']) \
                               / float(function_['stride']))+1
            out_width  = floor(float(input_size[2]-function_['k_size']) \
                               / float(function_['stride']))+1
        elif function_['padding']=='same':
            out_height = ceil(float(input_size[1]-function_['k_size']) \
                              / float(function_['stride']))+1
            out_width  = ceil(float(input_size[2]-function_['k_size']) \
                              / float(function_['stride']))+1
        return [input_size[0],int(out_height),int(out_width),input_size[3]]

    elif function_['type']=='relu':
        return input_size

    elif function_['type']=='norm':
        return input_size

    elif function_['type']=='fc':
        return [input_size[0],function_['output_size']]
        
def chain_size_crunch(input_size,funcs):
    # This applies calc_size_after() to every one of the funcs
    updated=input_size
    for f in funcs:
        updated=calc_size_after(updated,f)
    return updated