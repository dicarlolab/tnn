"""
Unicycle Class
"""

#                      STEP 0
#      ######          ######          ######
#       ####################################
#      ######          ######          ######


# Import all the future support libs
from __future__ import absolute_import, division

# Import TF and numpy - basic stuff
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell

# Import Unicycle-specific things
from unicycle_settings import *
from json_import import json_import
from construct_networkx import construct_G

import utility_functions
from utility_functions import fetch_node
from harbor import Harbor, Harbor_Dummy, Policy
from GenFuncCell import GenFuncCell
import imgs

# Import the system-related libs
import argparse
import os
import sys
from random import randint
from itertools import chain
if VERBOSE:
    from dbgr import dbgr_verbose as dbgr
else:
    from dbgr import dbgr_silent as dbgr


class Unicycle(object):
    def __init__(self):
        print 'Unicycle Initialized'

    def build(self):
        """
        The main execution routine file for the Universal Neural
        Interpretation and Cyclicity Engine (UNICYCLE)

        The way this works is the following:

        Step 1
        =======
         High-level description of the system is fed in via a JSON object-
         bearing file that is passed in as a command line argument when the
         script is run. Here we get the raw metadata from the JSON file and
         store it in a list of dictionaries, as well as a list of tuples of
         (from,to) links in the graph.

        Step 2
        =======
         Now we create a Network-X graph G for planning purposes. Store the
         nickname only as the pointer to the node to be instantiated, and then
         use this nickname to look up relevant node's metadata in the node
         dictionary list we acquired in step 1.

         Step 3
        =======
         BFS Search - currently unused, might slash this section soon.

        Step 4
        =======
         Using the Network-X graph G we create a parallel graph H and copy the
         main forward links into it.

            if target_node not on any path leading to source node:
                append (source_node, target_node) to forward_list
            NXGRAPH using forward_list -> H

        Step 5
        =======
         Once all the connections are made, we start the size calculation.
         This involves the Harbor of every one of the nodes (here the actual
         Tensors will be scaled or added or concatenated and the resulting
         Tensor will be used as input to the functional "conveyor belt").
         While the Harbor is the place where the actual resizing happens, we
         also have the Harbor-Master policy. This policy can be specified in
         the node metadata in the JSON, or if it isn't specified it can be
         inferred from the default settings (default subject to modification
         too).

         For every NODE:
         - Collect all non-feedback inputs, find their sizes, push list of
         sizes along with Harbor-Master policy into general Harbor-Master
         utility function to find ultimate size.
         - Create a reference dictionary for node metadata that has incoming
         inputs as keys and scaling values as values.
         - Calculate all final sizes for all nodes, use for feedback up and
         down the line.

        Step 6
        =======
         Tensor creation.

        Step 7
        =======
         Perform proper RNN unrolling of nodes within 1 time step. Thi
         cheating, as the RNN is essentially unrolled through a single
         memoized states it's parent and predecessor Cells are queried
         for, creating the illusion of a true RNN unroll. In reality,
         structure is preserved

        Let's kick ass
        """
        imgs.unicycle_logo()

        #                      STEP 1
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 1\n JSON Import and Parse\n======================')

        # Import NODES and LINKS from JSON
        nodes, links = json_import(dbgr=dbgr)

        #                      STEP 2
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 2\n Network-X Raw Build\n======')

        # Create NetworkX DiGraph G, find root nodes
        G, root_nodes = construct_G(links, dbgr=dbgr)

        #                      STEP 3
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 3\n BFS Dependency Parse\n======')
        dbgr('BFS Dependency Parse Temporarily Disabled')

        #                      STEP 4
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 4\n Clone Forward-Only Graph Creation\n======')

        # Create the forward-only DiGraph H
        H = construct_H(G, dbgr=dbgr)

        #                      STEP 5
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 5\n Input Size Calculation\n======')

        #                      STEP 6
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 6\n TF Node Creation\n========================')

        # with tf.Graph().as_default():
        #     # with tf.Graph().device(device_for_node):
        #     sess = tf.Session()
        #     with sess.as_default():

        #         # Initialize the first TF Placeholder to be pushed through
        #         # the Graph
        #         first_cell=GenFuncCell(harbor=node_harbors[first],
        #                                state_fs=[],
        #                                out_fs=[],
        #                                state_fs_kwargs=[],
        #                                out_fs_kwargs=[],
        #                                memory_kwargs={},
        #                                output_size=node_out_size[first],
        #                                state_size=node_state_size[first],
        #                                scope=first)

        # Repository of all the Tensor outputs for each Node in the TF Graph
        repo={}

        for i in node_input_touch:
            current_info=fetch_node(node)[0]

            #Initialize the TF Placeholder for this input
            this_input=GenFuncCell(harbor=node_harbors[i],
                                   state_fs=[], 
                                   out_fs=[], 
                                   state_fs_kwargs=[],
                                   out_fs_kwargs=[],
                                   memory_kwargs={},
                                   output_size=node_out_size[i], 
                                   state_size=node_state_size[i], 
                                   scope=i)

            repo[i]=this_input

        # Now, let's initialize all the nodes one-by-one
        for node in node_touch:
            current_info=fetch_node(node)[0]

            # Let's initiate TF Node:
            tf_node=GenFuncCell(harbor=node_harbors[node],
                               state_fs=\
                        [str(f['type']) for f in current_info['functions']], 
                               out_fs=[], 
                               state_fs_kwargs=\
        utility_functions.assemble_function_kwargs(current_info['functions'],
                                            node_harbors[node].desired_size,
                                            node),
                               out_fs_kwargs=[],
                               memory_kwargs={},
                               output_size=node_out_size[node], 
                               state_size=node_state_size[node], 
                               scope=str(node))
            repo[node]=tf_node


        # Now that the TF Nodes have been initialized, we build the Graph by
        # calling each Node with the appropriate inputs from the other Nodes:
        for node in node_touch:
            tf_node=repo[node]
            # Collect all the incoming inputs, including feedback:
            incoming_inputs_forward=H.predecessors(node)
            incoming_inputs_feedback=[i for i in G.predecessors(node) \
                                        if i not in incoming_inputs_forward]

            current_info=fetch_node(node)[0]

            # Assemble the correct inputs:
            # Inputs are {'nickname':Tensor}
            # First the forward inputs:
            inputs={i:repo[i].state for i in incoming_inputs_forward}
            # Then the backwards inputs:
            for i in incoming_inputs_feedback:
                inputs[i]=repo[i].state

            # Call the node with the correct inputs
            tf_node(inputs)

        # Emotional support
        dbgr(imgs.centaur())

        return repo[node_touch[-1]]


    def alexnet_demo_out(self, training_input=None, **kwargs):
        pass



#### Unroller should be almost entirely generic
#### Separate function!
## Unicycle 


    def unicycle_tfutils(self,training_data,**kwargs):
        m=self.alexnet_demo_out(training_data,**kwargs)
        return m.state, {'input': 'image_input_1',
                       'type': 'lrnorm',
                       'depth_radius': 4,
                       'bias': 1,
                       'alpha': 0.0001111,
                       'beta': 0.00001111}

if __name__=='__main__':
    print 'THIS\nIS\nA\nTEST\nUNICYCLE\nALEXNET\nINITIALIZATION'
    a=Unicycle()
    a.build()
    b=a.alexnet_demo_out()
