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
import tensorflow as tf

# Import the system-related libs
if VERBOSE:
    from dbgr import dbgr_verbose as dbgr
else:
    from dbgr import dbgr_silent as dbgr


class Unicycle(object):
    def __init__(self):
        print 'Unicycle Initialized'

    def build(self, json_file_name=None, dbgr=dbgr, train=False):
        """
        The main build routine for the Universal Neural
        Interpretation and Cyclicity Engine (UNICYCLE)

        The way this works is the following:

        Step 1
        =======
         High-level description of the system is fed in via a JSON object-
         bearing file that is passed in as a command line argument when the
         script is run (use `-f <json_file_name_here.json>` argument in the
         command line). Here we get the raw metadata from the JSON file and
         store it in a list of dictionaries, as well as a list of tuples of
         (from,to) links in the graph.

        Step 2
        =======
         Now we create a Network-X graph G for planning purposes. Store the
         nickname only as the pointer to the node to be instantiated, and then
         use this nickname to look up relevant node's metadata in the node
         dictionary list we acquired in step 1.

         Using the Network-X graph G we also find all edges that are feedback
         edges and mark them accordingly. This Network-X DiGraph is what we
         pass around and populate with all the different objects and metadata
         about the different nodes (TF objects as well as Python objects).

        Step 3
        =======
         Once all the connections are made, we start the size calculation.
         This involves the Harbor of every one of the nodes (here the actual
         Tensors will be scaled or added or concatenated and the resulting
         Tensor will be used as input to the functional "conveyor belt").
         While the Harbor is the place where the actual resizing happens, we
         also have the Harbor Policy. The Policy can be specified in the node
         metadata in the JSON, or if it isn't specified it can be inferred
         from the default settings (default subject to modification too). The
         Policy object has specific transformations defined - these trans-
         formations are to be applied by Harbor on the incoming inputs, and
         so whereas the Harbor is a clean generalized framework for the input
         to be worked on, the Policy is the object that contains the nitty-
         gritty details of what exactly should happen where. Harbors can
         receive custom functions from the user by receiving a user-created
         Policy.

        Step 4
        =======
         Once all the sizes for all the nodes have been calculated, we
         initialize all of them one-by-one in order of dependence (input
         first, then GenFuncCells right after the input, etc). Initialization
         allows us to create TF nodes that we save for each GenFuncCell in the
         Network-X graph, and also allows us to link up different GenFuncCells
         together in the TF graph using Harbors. Here the recurrent (feedback)
         edges of the graph are also processed, so the final TF Graph is a
         properly connected and resized Graph.

         We then return the Network-X object containing all of this info.



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
        G = initialize_nodes(G, train=train)

        # Emotional support
        dbgr(imgs.centaur())

        return G

    def __call__(self, input_sequence, G, dbgr=dbgr):
        """
        When a built and initialized Unicycle instance is called, a custom-
        written unroller 'walks' through the TF graph, and pushes the input
        data from the beginning to the end through time and space. For the
        purposes of Unicycle, we unroll through time, and at every time step
        we unroll all of the nodes and push the input (or the output from the
        previous time point) through. The function returns the TF node of the
        output GenFuncCell.

        """

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

    def build_and_output(self,
                         inputs=[],
                         json_file_name=None,
                         train=False,
                         **kwargs):
        G = self.build(json_file_name=json_file_name,
                       train=train)
        last_ = self(inputs, G)
        return last_


def unicycle_tfutils(inputs, **kwargs):
    """
    This function is not specific to any architecture, the particulars of the
    design of the network are to be specified in a JSON file and saved in the
    'json' folder in the root of tconvnet. To point Unicycle to the desired
    JSON file just use the `-f <json_file_name_here.json>` command line
    argument.
    We initialize the Unicycle graph with the AlexNet architecture, then we
    push the `inputs` through and receive the state of the output node
    """
    m = Unicycle()
    o = m.build_and_output(inputs, **kwargs)
    return o.get_state(), {'input': 'image_input_1',
                           'type': 'lrnorm',
                           'depth_radius': 4,
                           'bias': 1,
                           'alpha': 0.0001111,
                           'beta': 0.00001111}


def mnist_tfutils(inputs, **kwargs):
    """
    This function is specific to MNIST (JSON file in build_and_output params
    below).
    We initialize the Unicycle graph with the AlexNet architecture, then we
    push the `inputs` through and receive the state of the output node
    """
    m = Unicycle()
    o = m.build_and_output(inputs,
                           json_file_name='sample_mnist.json',
                           **kwargs)
    return o.get_state(), {'input': 'image_input_1',
                           'type': 'lrnorm',
                           'depth_radius': 4,
                           'bias': 1,
                           'alpha': 0.0001111,
                           'beta': 0.00001111}


def mnist_benchmark(inputs, **kwargs):
    """
    This function is specific to MNIST (JSON file in build_and_output params
    below).
    We initialize the Unicycle graph with the AlexNet architecture, then we
    push the `inputs` through and receive the state of the output node
    """

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.global_variables_initializer().run()}

    IMAGE_SIZE = 28
    NUM_CHANNELS = 1
    NUM_LABELS = 10
    SEED = 66478  # Set to None for random seed.

    input_shape = inputs['images'].get_shape().as_list()
    inputs = tf.reshape(inputs['images'], [input_shape[0], int(input_shape[2]**0.5), int(input_shape[2]**0.5), NUM_CHANNELS])

    BATCH_SIZE = input_shape[0]

    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=SEED, dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=tf.float32))

    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(inputs,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    
    return tf.matmul(hidden, fc2_weights) + fc2_biases, {'input': 'image_input_1',
                           'type': 'lrnorm',
                           'depth_radius': 4,
                           'bias': 1,
                           'alpha': 0.0001111,
                           'beta': 0.00001111}


def alexnet_tfutils(inputs, **kwargs):
    """
    This function is specific to AlexNet (JSON file in build_and_output params
    below).
    We initialize the Unicycle graph with the AlexNet architecture, then we
    push the `inputs` through and receive the state of the output node
    """
    m = Unicycle()
    o = m.build_and_output(inputs,
                           json_file_name='sample_alexnet.json',
                           **kwargs)
    return o.get_state(), {'input': 'image_input_1',
                           'type': 'lrnorm',
                           'depth_radius': 4,
                           'bias': 1,
                           'alpha': 0.0001111,
                           'beta': 0.00001111}


def densenet_tfutils(inputs, **kwargs):
    """
    This function is specific to AlexNet (JSON file in build_and_output params
    below).
    We initialize the Unicycle graph with the AlexNet architecture, then we
    push the `inputs` through and receive the state of the output node
    """
    m = Unicycle()
    o = m.build_and_output(inputs,
                           json_file_name='sample_densenet.json',
                           **kwargs)
    return o.get_state(), {'input': 'image_input_1',
                           'type': 'lrnorm',
                           'depth_radius': 4,
                           'bias': 1,
                           'alpha': 0.0001111,
                           'beta': 0.00001111}


# if __name__ == '__main__':
#     print 'THIS\nIS\nA\nTEST\nUNICYCLE\nALEXNET\nINITIALIZATION'
#     a = Unicycle()
#     b = a.mnist_demo_out(['', '', '', '', '', '', '', '', '', ''])
