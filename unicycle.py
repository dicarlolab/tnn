"""
The main execution routine file for the 
Universal Neural Interpretation and Cyclicity Engine (UNICYCLE)

The way this works is the following:

Step 1
=======
 High-level description of the system is fed in via a JSON object-bearing
 file that is passed in as a command line argument when the script is run.
 Here we get the raw metadata from the JSON file and store it in a list of 
 dictionaries, along with all the information about the node

Step 2
=======
 No we create an Network-X graph G for planning purposes. Store the nickname 
 only as the pointer to the node to be instantiated, and then use this 
 nickname to look up relevant node's metadata in the node dictionary list we 
 acquired in step one.

Step 3
=======
 Using the Network-X graph G we will find the longest simple path from start 
 to finish. For now we will use the notation 
    max(nx.all_simple_paths(G,start,end), key=len)
 Returns a list of node names (a path).

Step 4
=======
 Using the Network-X graph G and the newly acquired longest path from step 3 
 we create a parallel graph H and copy the main "spine" into it. Then, iterate 
 through the "spine" nodes and look up their connections in graph G.

Step 5
=======
 For each node on the "spine", if the incoming link is from an ancestor then 
 we add it as-is. If, however, the incoming link is not from an ancestor (i.e. 
 incoming from the future), add the link to the node with a ~ attribute in the
 metadata of the node. This is done to let the system know down the line that 
 the past state needs to be accessed. 
 
Step 6
=======
 Once all the connections are made, we start the size calculation. This 
 involves the Harbor of every one of the nodes (here the actual Tensors will 
 be scaled or added or concatenated and the resulting Tensor will be used as 
 input to the functional "conveyor belt"). While the Harbor is the place where 
 the actual resizing happens, we also have the Harbor-Master policy. This 
 policy can be specified in the node metadata in the JSON, or if it isn't 
 specified it can be inferred from the default settings (default subject to 
 modification too). 

 For every NODE:
 - Collect all non-feedback inputs, find their sizes, push list of sizes along 
 with Harbor-Master policy into general Harbor-Master utility function to find 
 ultimate size. 
 - Create a reference dictionary for node metadata that has incoming inputs as 
 keys and scaling values as values. 
 - Calculate all final sizes for all nodes, use for feedback up and down the 
 line.


Step 7
=======
 Perform proper RNN unrolling of nodes within 1 time step. This is almost
 cheating, as the RNN is essentially unrolled through a single time step but
 memoized states it's parent and predecessor Cells are queried and accounted
 for, creating the illusion of a true RNN unroll. In reality, DAG forward 
 structure is preserved

Let's kick ass
"""



            #                      STEP 0                           $$$$$$\  
            #             INITIALIZATION AND SETUP                 $$$ __$$\ 
            #      ######          ######          ######          $$$$\ $$ |
            #       ####################################           $$\$$\$$ |
            #      ######          ######          ######          $$ \$$$$ |
            #                                                      $$ |\$$$ |
            #    Prepare the environment and import all the        \$$$$$$  /
            #     necessary supplementary libraries and/or          \______/ 
            #       classes from other files and repos.                   
# Import all the future support libs
from __future__ import absolute_import, division

verbose=True

def dbgr(_string='', leave_line_open=0):
    if leave_line_open:
        if verbose: print _string,
    else:
        if verbose: print _string

print(r"""
        Welcome to

$$\   $$\ $$\   $$\ $$$$$$\  $$$$$$\ $$\     $$\  $$$$$$\  $$\       $$$$$$$$\ 
$$ |  $$ |$$$\  $$ |\_$$  _|$$  __$$\\$$\   $$  |$$  __$$\ $$ |      $$  _____|
$$ |  $$ |$$$$\ $$ |  $$ |  $$ /  \__|\$$\ $$  / $$ /  \__|$$ |      $$ |      
$$ |  $$ |$$ $$\$$ |  $$ |  $$ |       \$$$$  /  $$ |      $$ |      $$$$$\    
$$ |  $$ |$$ \$$$$ |  $$ |  $$ |        \$$  /   $$ |      $$ |      $$  __|   
$$ |  $$ |$$ |\$$$ |  $$ |  $$ |  $$\    $$ |    $$ |  $$\ $$ |      $$ |      
\$$$$$$  |$$ | \$$ |$$$$$$\ \$$$$$$  |   $$ |    \$$$$$$  |$$$$$$$$\ $$$$$$$$\ 
 \______/ \__|  \__|\______| \______/    \__|     \______/ \________|\________|

        *The Universal Neural Interpretation and Cyclicity Engine

""")

dbgr('Verbose Mode is ON')
dbgr()

dbgr('Starting Basic Library Import...', 1)

# Import TF and numpy - basic stuff
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell


# Import the system-related libs
import argparse
import os
import sys
from random import randint

# Import Network-X and JSON support libraries
import networkx as nx
import json

dbgr('done!')
dbgr()



            #                      STEP 1                              $$\   
            #               JSON IMPORT AND PARSE                    $$$$ |  
            #      ######          ######          ######            \_$$ |  
            #       ####################################               $$ |  
            #      ######          ######          ######              $$ |  
            #                                                          $$ |  
            #   High-level description of the system is fed in       $$$$$$\ 
            #    via a JSON object-bearing file that is passed       \______|
            #    in as a command line argument when the script 
            #                     is executed
dbgr('======\nSTEP 1\n JSON Import and Parse\n======================')
if len(sys.argv)<2:
    # raise Exception('You need to specify the name of JSON settings file!')
    dbgr('No JSON settings file specified! Scanning current directory...')
    for f in os.listdir('.'):
        if f.endswith('.json'):
            dbgr('Using first discovered JSON file in current path: %s'\
                                                                %(f.upper()))
            json_file_name=f
            break
else:
    # Fetch the name of the JSON file from the command line args
    json_file_name=sys.argv[1]

dbgr('Using JSON file %s for settings import...'%(json_file_name.upper()),1)

with open('./'+json_file_name) as data_file:    
    json_data = json.load(data_file)

dbgr('done!')
dbgr()

dbgr('Checking the integrity of the JSON file...')

assert 'nodes' in json_data, "NODES field not in JSON file!"
assert len(json_data['nodes'])>0, "No NODES in the JSON file!"
assert 'forward' in json_data, "FORWARD link field not in JSON file!"
if len(json_data['forward'])==0: print 'Warning: FORWARD links empty!'
assert 'backward' in json_data, "BACKWARD link field not in JSON file!"
if len(json_data['backward'])==0: print 'Warning: BACKWARD links empty!'

nodes=json_data['nodes']
forward=json_data['forward']
backward=json_data['backward']

first=str(nodes[0]['nickname'])
final=str(nodes[-1]['nickname'])

# Let's make sure that pooling doesn't appear on its own in the Cell:
assert not any( [all([f['type']=='relu' for f in n['functions']]) \
            for n in nodes]), 'ReLu cannot appear by itself in a Cell!'

dbgr('done!')
dbgr()


            #                      STEP 2                           $$$$$$\  
            #                KWARGS PREPARATION                    $$  __$$\ 
            #      ######          ######          ######          \__/  $$ |
            #       ####################################            $$$$$$  |
            #      ######          ######          ######          $$  ____/ 
            #                                                      $$ |      
            #     We are going to first iterate through all        $$$$$$$$\ 
            #  the nodes we extracted from the JSON file and       \________|
            #   apply a series of additional steps to generate
            #   TF-function-ready kwarg dictionaries with all the
            #                 correct arguments          
dbgr('======\nSTEP 2\n Keyword Argument Prep\n======================')

# Create repository of all the nodes, addressed by name
repo={}
for node in nodes:
    name=node['nickname']
    assert name not in repo, 'Naming conflict! Name %s already exists'%(name)
    # The dictionary contains a list of two items:
    #   0. The final kwargs dictionary for that node
    #   1. The incoming Cells and their sizes
    #   2. The scaling factor necessary for every one of the inputs
    repo[name]=[{},[],[]]

def createKwargsConv(fn, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict={}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type']=='conv', 'Type has to be CONV, but is %s'%(fn['type'])
    filter_tensor=tf.get_variable('filter_tensor',
                                  fn['filter_size']\
                                    +[properlyScaled(repo[fn[name]][1])]\
                                    +[fn['num_filters']],
                                initializer=tf.random_uniform_initializer(0,1)
                                 )
    out_dict['filter']=filter_tensor
    out_dict['strides']=fn['stride']
    out_dict['padding']=fn['padding']
    out_dict['name']=str(name) if isinstance(name, basestring) \
                               else 'conv_%s'%(str(randint(1,1000)))

    return out_dict

def properlyScaled(list_of_incoming_tuples):
    # Do some magic and figure out the biggest input size, create internal
    # structure to reshape all the input to that size and concatenate in
    # terms of channels
    max_size_so_far=0
    for incoming_tuple in list_of_incoming_tuples:
        node_size=incoming_tuple[1][0]
        if node_size>max_size_so_far:
            max_size_so_far=node_size
    for incoming_tuple in list_of_incoming_tuples:
        scaling_factor=max_size_so_far//incoming_tuple[1][0]
        # Not sure how to do this yet, sketchy:
        resized = tf.image.resize_images(input_tensor, [new_height, new_width])
        repo[incoming_tuple[0]][2].append(resized)

def createKwargsMaxpool(fn, name):
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
    return out_dict

def createKwargsRelu(fn, name):
    # This function takes the abstract JSON representation and prepares the
    # correct function-ready kwarg collection
    out_dict={}
    assert 'type' in fn, 'Type of function not in function!'
    assert fn['type']=='relu', 'Type has to be RELU, but is %s'%(fn['type'])

    out_dict['name']=str(name) if isinstance(name, basestring) \
                               else 'relu_%s'%(str(randint(1,1000)))
    return out_dict

def createKwargsNorm(fn, name):
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
    return out_dict

# dbgr('Creating a Network-X Graph...',1)
# # Create Graph
# NXG = nx.DiGraph()
# dbgr('done!')
# dbgr()


# nlayers = len(nodes)
# NXG.add_node("image_input_1", cell=None, name='input')
# names = []
# for node, layer in enumerate(layers):
#     node = str(node + 1)
#     cell = layer()  # initialize cell
#     NXG.add_node(node, cell=cell, name=cell.scope)
#     NXG.add_edge(str(int(node)-1), node)

# #  adjacent layers
# # NXG.add_edges_from([(names[i], names[j]) for i,j in bypasses])  # add bypass connections
# NXG.add_edges_from([(str(i), str(j)) for i,j in bypasses])
# # print(NXG.nodes())

# # check that bypasses don't add extraneous nodes
# if len(NXG) != nlayers + 1:
#     import pdb; pdb.set_trace()
#     raise ValueError('bypasses list created extraneous nodes')

# dbgr('Network-X Model Functionality Temporarily Disabled')

dbgr()

                                                                            
            #                      STEP 3                           $$$$$$\  
            #             GENFUNCCELL INSTANTIATION                $$ ___$$\ 
            #      ######          ######          ######          \_/   $$ |
            #       ####################################             $$$$$ / 
            #      ######          ######          ######            \___$$\ 
            #                                                      $$\   $$ |
            #   The NODES of the Network-X model are converted     \$$$$$$  |
            #   into appropriate General Functional Cells, and      \______/ 
            #  their internal metadata is used to populate the              
            #    functional space inside of each of the cells
dbgr('======\nSTEP 3\n GenFuncCell Instantiation\n==========================')


# For every node in the JSON description, create an instance of GenFuncCell
# and populate it with the proper functions, as necessary.

tf_cells={}

for node in nodes:
    # For every node we first collect the function information
    node_type=str(node['type'])
    node_functions=node['functions']
    to_be_passed_in_state=[]
    to_be_passed_in_state_kwargs=[]
    for f in node_functions:
        f_type=f['type']
        to_be_passed_in_state.append(f_type)
        temp_kwarg={}
        if f_type=='conv':
            temp_kwarg=createKwargsConv(f, node['nickname'])
        elif f_type=='maxpool':
            temp_kwarg=createKwargsMaxpool(f, node['nickname'])
        elif f_type=='relu':
            temp_kwarg=createKwargsRelu(f, node['nickname'])
        elif f_type=='norm':
            temp_kwarg=createKwargsNorm(f, node['nickname'])
        to_be_passed_in_state_kwargs.append(temp_kwarg)

    tf_cells[node['nickname']]=GenFuncCell(to_be_passed_in_state,
                                           [],
                                           to_be_passed_in_state_kwargs,
                                           {},
                                           {},
                                           scope=node['nickname'])

dbgr()

            #                      STEP 4                          $$\   $$\ 
            #             PROGRESS/BYPASS/FEEDBACK                 $$ |  $$ |
            #      ######          ######          ######          $$ |  $$ |
            #       ####################################           $$$$$$$$ |
            #      ######          ######          ######          \_____$$ |
            #                                                            $$ |
            #         The EDGES of the Network-X model                   $$ |
            #    (progress/bypass/feedback) are converted to             \__|
            #      progress/bypass/feedback connections in                  
            #       the TF graph. This is done by adding                    
            #    pointers/references from every TF object to                
            #    every other object that it receives inputs                 
            #   from, from both the previous and the current                
            #                   time steps.                                 
            #                                                               
            #   Proper sizes for the input/state/output sizes               
            #      of all the TF Cells are calculated and                   
            #                  accounted for.                               
dbgr('======\nSTEP 4\n Progress/Bypass/Feedback\n=========================')


for link in forward:
    repo[link['to']][1].append(link['from'])



dbgr()

            #                      STEP 5                          $$$$$$$\  
            #               CUSTOM RNN UNROLLING                   $$  ____| 
            #      ######          ######          ######          $$ |      
            #       ####################################           $$$$$$$\  
            #      ######          ######          ######          \_____$$\ 
            #                                                      $$\   $$ |
            #    Proper RNN unrolling of nodes within 1 time       \$$$$$$  |
            #    step is performed. This is almost cheating,        \______/ 
            #     as the RNN is essentially unrolled through                
            #    a single time step but memoized states it's                
            #     parent and predecessor Cells are queried                  
            #   and accounted for, creating the illusion of a               
            #     true RNN unroll. In reality, DAG forward                  
            #             structure is preserved                            
dbgr('======\nSTEP 5\n Custom RNN Unrolling\n=====================')
dbgr('Derp Derp Derp')
dbgr()