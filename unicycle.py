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
    list(nx.bfs_edges(G,first))
 Returns a list of edges.

Step 4
=======
 Using the Network-X graph G we create a parallel graph H and copy the main 
 forward links into it. 
    [(fr,to) for (fr,to) in G.edges()
        if not any([to in i for i in nx.all_simple_paths(G,first,fr)])]
 Then, iterate through the nodes in H and look up the connections in graph G.

 For each node on the "spine", if the incoming link is from an ancestor then 
 we add it as-is. If, however, the incoming link is not from an ancestor (i.e. 
 incoming from the future), add the link to the node with a ~ attribute in the
 metadata of the node. This is done to let the system know down the line that 
 the past state needs to be accessed. 
 
Step 5
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

Step 6
=======
 Tensor creation.

Step 7
=======
 Perform proper RNN unrolling of nodes within 1 time step. This is almost
 cheating, as the RNN is essentially unrolled through a single time step but
 memoized states it's parent and predecessor Cells are queried and accounted
 for, creating the illusion of a true RNN unroll. In reality, DAG forward 
 structure is preserved

Let's kick ass
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


# Import the system-related libs
import argparse
import os
import sys
from random import randint
from itertools import chain

# Import Network-X and JSON support libraries
import networkx as nx
import matplotlib.pyplot as plt
import json

# Import Unicycle-specific things
from unicycle_settings import *
import utility_functions
from harbor import Harbor, Harbor_Dummy, Policy
from GenFuncCell import GenFuncCell


class Unicycle(object):
    def __init__(self):
        print 'Unicycle Initialized'

    def alexnet_demo_out(self, training_input=None, **kwargs):

        verbose=True

        def dbgr(_string='', leave_line_open=0, newline=True):
            if leave_line_open:
                if verbose: print _string,
            else:
                if verbose: 
                    if newline:
                        print _string,'\n'
                    else: print _string

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


                    #                      STEP 1                      
                    #      ######          ######          ######      
                    #       ####################################       
                    #      ######          ######          ######      



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

        dbgr('Checking the integrity of the JSON file...')

        assert 'nodes' in json_data, "NODES field not in JSON file!"
        assert len(json_data['nodes'])>0, "No NODES in the JSON file!"
        assert 'links' in json_data, "LINKS link field not in JSON file!"
        if len(json_data['links'])==0: print 'Warning: LINKS empty!'
        # assert 'backward' in json_data, "BACKWARD link field not in JSON file!"
        # if len(json_data['backward'])==0: print 'Warning: BACKWARD links empty!'

        nodes=json_data['nodes']
        links=json_data['links']
        # backward=json_data['backward']

        # By old convention, the first node is literally the first node in JSON
        # Deprecated
        # first=str(nodes[0]['nickname'])
        final=str(nodes[-1]['nickname'])

        # Let's make sure that pooling doesn't appear on its own in the Cell:
        assert not any( [all([f['type']=='relu' for f in n['functions']]) \
                    for n in nodes]), 'ReLu cannot appear by itself in a Cell!'

        dbgr('done!')



                    #                      STEP 2                     
                    #      ######          ######          ######      
                    #       ####################################       
                    #      ######          ######          ######      



        dbgr('======\nSTEP 2\n Network-X Raw Build\n======================')


        # Convert list of link dictionaries to list of link tuples to be fed into 
        # Network-X
        links_tuples=[(str(i['from']),str(i['to'])) for i in links]

        dbgr('Building Network-X Raw DiGraph...')
        # We create a Network-X DiGraph G and populate it using the links.
        G=nx.DiGraph()
        G.add_edges_from(links_tuples)
        # G.add_edge('fc_8','conv_3')

        dbgr('Network-X Raw Graph created! Nodes: ',newline=False)
        dbgr('    '+'\n    '.join(sorted(G.nodes())))

        # Now we find the starting nodes - these will be nodes without predecessors,
        # nodes that are either 
        #   a) input nodes with placeholders
        #   b) bias nodes
        # The list of these nodes will be collected in the `root_nodes` variable
        root_nodes=[i for i in G.nodes() if len(G.predecessors(i))==0]

        # Draw the Graph
        # dbgr('Drawing graph')
        # pos=nx.circular_layout(G)
        # labels={str(i):str(i).upper() for i in G.nodes()}
        # nx.draw(G,pos)
        # nx.draw_networkx_labels(G,pos,labels)
        # dbgr('Close the Matlab preview window to continue')
        # plt.show()
        # dbgr()


                                                                                    
                    #                      STEP 3                    
                    #      ######          ######          ######      
                    #       ####################################       
                    #      ######          ######          ######      



        dbgr('======\nSTEP 3\n BFS Dependency Parse\n====================')

        dbgr('BFS Dependency Parse Temporarily Disabled')

        # # Find the longest simple path through the Graph, and save it as "spine"
        # dbgr('Finding all the forward dependeny links in the Graph using BFS...',1)
        # # spine=max([i for i in nx.all_simple_paths(G,first,final)], key=len)
        # forward_bfs=list(nx.bfs_edges(G,first))
        # dbgr('done!')
        # dbgr('Links: ',newline=False)
        # dbgr('    '+'\n    '.join([str(i) for i in forward_bfs]))



                    #                      STEP 4                  
                    #      ######          ######          ######      
                    #       ####################################       
                    #      ######          ######          ######  



        dbgr('======\nSTEP 4\n Clone Forward-Only Graph Creation\n==================')

        dbgr('Finding all non-ancestral dependeny links in the Graph...',1)
        # Gotta love string comprehensions :)
        # Add only those edges that lead forward (the target is not an ancestor)
        forward=[(fr,to) for (fr,to) in G.edges() \
                    if not any([to in i for i in \
                                    # All paths lead to Rome! (from all roots to `to`)
                                list(chain(*[nx.all_simple_paths(G,first,fr) \
                                            for first in root_nodes]))
                              ])
                ]

        dbgr('done!')
        dbgr('Links: ',newline=False)
        dbgr('    '+'\n    '.join([str(i) for i in forward]))

        # Let's create a forward-only copy of G --> H
        H=nx.DiGraph(forward)
        dbgr('Nodes of forward-only Graph: ',newline=False)
        dbgr('    '+'\n    '.join(sorted(H.nodes())))

        # Print all root nodes
        dbgr('Root nodes: ',newline=False)
        dbgr('    '+'\n    '.join(sorted(root_nodes)))



                    #                      STEP 5
                    #      ######          ######          ######      
                    #       ####################################       
                    #      ######          ######          ######     



        dbgr('======\nSTEP 5\n Input Size Calculation\n=====================')

        # Helper function to help with fetching node data from the big dump
        def fetch_node(nickname='no_nickname_given', node_storage=nodes, **kwargs):
            if len(kwargs)>0:
                dict_to_iterate=kwargs
            else:
                dict_to_iterate={'nickname':nickname}
            matching=[]
            for k,v in dict_to_iterate.items():
                contains_this_val=[i for i in [ii for ii in node_storage if k in ii] \
                                              if i[k]==v]
                matching=matching+[i for i in contains_this_val if i not in matching]
            return matching

        dbgr('Starting the node traversal for size calculation. If this step hangs \
        check the node_out_size validation loop in <Step 5>.')

        # This is a hash table that looks at whether a particular node has been
        # traversed or not (if it's size has been calculated)
        node_out_size={i:None for i in H.nodes()}
        node_state_size={i:None for i in H.nodes()}
        node_harbors={}
        # Create a list to store the correct initialization order for the nodes:
        node_input_touch=[]
        node_bias_touch=[]
        node_touch=[]

        # Calculate the input sizes for Placeholders and Biases
        dbgr('Calculating Input sizes...',1)

        # We assume that input image size has been specified
        input_nodes=fetch_node(type='placeholder')
        bias_nodes=fetch_node(type='bias')

        # Make sure all the batch sizes are the same:
        assert len(set([i['batch_size'] for i in input_nodes]))==1, 'Batches differ!'
        BATCH_SIZE=input_nodes[0]['batch_size']

        # Make sure the functions are there, and that the output sizes are specified
        # Then, specify the node state and output sizes
        # Then, add a Dummy Harbor for that input
        # Then, add to node_touch for proper ordering
        for i in input_nodes:
            this_name=i['nickname']
            assert 'functions' in i, 'Input node %s has no functions!'%(this_name)
            assert 'output_size' in i['functions'][0], 'Input node %s has no output\
                                                    size specified!'%(this_name)
            # [batch, height, width, channels], batch is usually set to None
            node_out_size[this_name]=[BATCH_SIZE]+i['functions'][0]['output_size']
            node_state_size[this_name]=node_out_size[this_name][:]
            node_harbors[this_name]=Harbor_Dummy(node_out_size[this_name],input_=True)
            node_input_touch.append(this_name)

        # Same for bias terms
        # Make sure the functions are there, and that the output sizes are specified
        # Then, specify the node state and output sizes
        # Then, add a Dummy Harbor for that input
        # Then, add to node_touch for proper ordering
        for i in bias_nodes:
            this_name=i['nickname']
            assert 'functions' in i, 'Bias node %s has no functions!'%(this_name)
            assert 'output_size' in i['functions'][0], 'Bias node %s has no output\
                                                    size specified!'%(this_name)
            # [batch, height, width, channels], batch is usually set to None
            node_out_size[this_name]=[BATCH_SIZE]+i['functions'][0]['output_size']
            node_state_size[this_name]=node_out_size[this_name][:]
            node_harbors[this_name]=Harbor_Dummy(node_out_size[this_name])
            node_bias_touch.append(this_name)

        dbgr('done!')

        dbgr('Starting Node Output Sizes:',newline=False)
        for i in sorted(node_out_size):
            dbgr('   '+i.ljust(max([len(j) for j in sorted(node_out_size)])) \
                 +':'+str(node_out_size[i]),newline=False)
        dbgr(newline=False)

        # Now let's write a loop that check the dependencies of nodes to make sure
        # everything preceding a node has been calculated:
        while not all(node_out_size.values()):
            for node in node_out_size:
                if node_out_size[node]:
                    # Current node has been calculated already, skip!
                    continue
                else:
                    if all([node_out_size[i] for i in H.predecessors(node)]):
                        dbgr('Counting up the sizes for node %s'%(node), 1)
                        # All the predecessors have been traversed, we can now proceed

                        # First, gather all the necessary info about the current node:
                        current_info=fetch_node(node)[0]

                        # # Change the past_1_present_0 value to length from root
                        # # Then, gather the sizes of all incoming nodes into a dict:
                        # # {'nickname':(size,past_1_present_0)}
                        # incoming_sizes={}
                        # for pred in G.predecessors(node):
                        #     if pred in H.predecessors(node):
                        #         # If the predecessor is in H then it's forward only
                        #         incoming_sizes[pred]=(node_out_size[pred],0)
                        #     else:
                        #         # If it's not in H, then it's a past feedback link
                        #         incoming_sizes[pred]=(node_out_size[pred],1)

                        # Then, gather the sizes of all incoming nodes into a dict:
                        # {'nickname':(size,all_simple_paths)}
                        incoming_sizes={}
                        for pred in G.predecessors(node):
                            incoming_sizes[pred]=(node_out_size[pred],
                                    list(chain(*[nx.all_simple_paths(H,st,pred) 
                                                for st in root_nodes])))

                        # Create a Policy instance
                        current_policy=Policy()

                        # Create a Harbor instance
                        node_harbors[node]=Harbor(incoming_sizes,
                                          policy=current_policy,
                                          node_name=node)

                        # Extract the desired_size from Harbor:
                        desired_size=node_harbors[node].get_desired_size()

                        # Find the size of the state:
                        current_state_size=utility_functions.chain_size_crunch(\
                                               desired_size,current_info['functions'])

                        # Find the size of the output (same as state for now):
                        current_out_size=current_state_size[:]

                        # Update node_out_size and node_state_size
                        node_out_size[node]=current_out_size[:]
                        node_state_size[node]=current_state_size[:]

                        # Add the current node to the node_touch list for further TF
                        # Tensor creation order
                        node_touch.append(node)

                        dbgr(' ... done!',newline=False)

                        # break the for loop after modifying the node_out_size dict
                        break
                    else:
                        # There are other nodes that need to be calculated, skip
                        continue
        dbgr()
        dbgr('Final Node Output Sizes:',newline=False)
        for i in sorted(node_out_size):
            dbgr('   '+i.ljust(max([len(j) for j in sorted(node_out_size)])) \
                 +':'+str(node_out_size[i]),newline=False)
        dbgr(newline=False)

        dbgr('\nAll sizes calculated!')



                    #                      STEP 6
                    #      ######          ######          ######      
                    #       ####################################       
                    #      ######          ######          ######      

        dbgr('======\nSTEP 6\n TF Node Creation\n========================')


        # with tf.Graph().as_default():
        #     # with tf.Graph().device(device_for_node):    
        #     sess = tf.Session()
        #     with sess.as_default():

        #         # Initialize the first TF Placeholder to be pushed through the Graph
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

        for i in node_input_touch+node_bias_touch:
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


        return repo[node_touch[-1]]

    def unicycle_tfutils(self,training_data,**kwargs):
        m=self.alexnet_demo_out(training_data,**kwargs)
        return m.state, {'input': 'image_input_1',
                       'type': 'lrnorm',
                       'depth_radius': 4,
                       'bias': 1,
                       'alpha': 0.0001111,
                       'beta': 0.00001111}

if __name__=='__main__':
    a=Unicycle()
    b=a.alexnet_demo_out()
