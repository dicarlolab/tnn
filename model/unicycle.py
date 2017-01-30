"""
Unicycle Class
"""

#                      STEP 0
#      ######          ######          ######
#       ####################################
#      ######          ######          ######


# Import all the future support libs
from __future__ import absolute_import, division

# Import Unicycle-specific things
from unicycle_settings import VERBOSE
from json_import import json_import
from construct_networkx import construct_G
from node_sizing import all_node_sizes
from initialize_nodes import initialize_nodes
from unroller import unroller_call
from utility_functions import fetch_node
import imgs

# Import the system-related libs
if VERBOSE:
    from dbgr import dbgr_verbose as dbgr
else:
    from dbgr import dbgr_silent as dbgr


class Unicycle(object):
    def __init__(self):
        print 'Unicycle Initialized'

    def build(self, json_file_name=None, dbgr=dbgr):
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

         Using the Network-X graph G we find all edges that are feedback and
         mark them as such with a 'feedback' attribute

        Step 3
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
        nodes, links = json_import(filename=json_file_name, dbgr=dbgr)

        #                      STEP 2
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 2\n Network-X Raw Build\n======')

        # Create NetworkX DiGraph G, find root nodes
        G = construct_G(nodes=nodes, links=links, dbgr=dbgr)

        #                      STEP 3
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 3\n Input Size Calculation\n======')

        # Calculate the sizes of all of the nodes here
        # node_out_size, node_state_size, node_harbors, node_input_touch, \
        #     node_touch = all_node_sizes(G, H, nodes, dbgr=dbgr)
        G = all_node_sizes(G, dbgr=dbgr)

        #                      STEP 4
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nSTEP 4\n TF Node Creation\n========================')

        # Initialize all the nodes:
        G = initialize_nodes(G)

        # Emotional support
        dbgr(imgs.centaur())

        return G

    def __call__(self, input_sequence, G, dbgr=dbgr):
        #                      STEP 7
        #      ######          ######          ######
        #       ####################################
        #      ######          ######          ######

        dbgr('======\nTF Unroller\n========================')

        G, last = unroller_call(
            input_sequence,
            G,
            fetch_node(output_layer=True, graph=G)[0]['tf_cell'])

        return last

    def alexnet_demo_out(self, inputs=[], **kwargs):
        G = self.build()
        last_ = self(inputs, G)
        return last_

    def mnist_demo_out(self, inputs=[], **kwargs):
        G = self.build(json_file_name='sample_mnist.json')
        last_ = self(inputs, G)
        return last_


def unicycle_tfutils(inputs, **kwargs):
    m = Unicycle()
    o = m.mnist_demo_out(inputs, **kwargs)
    return o.get_state(), {'input': 'image_input_1',
                           'type': 'lrnorm',
                           'depth_radius': 4,
                           'bias': 1,
                           'alpha': 0.0001111,
                           'beta': 0.00001111}


if __name__ == '__main__':
    print 'THIS\nIS\nA\nTEST\nUNICYCLE\nALEXNET\nINITIALIZATION'
    a = Unicycle()
    b = a.mnist_demo_out(['', '', '', '', '', '', '', '', '', ''])
