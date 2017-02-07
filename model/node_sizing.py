from dbgr import dbgr_silent
from unicycle_settings import BATCH_SIZE
from utility_functions import fetch_node
import utility_functions
from harbor import Harbor, Policy
from itertools import chain
import networkx as nx


def all_node_sizes(G, dbgr=dbgr_silent):

    G = root_node_size_find(G, dbgr=dbgr)

    G = all_node_size_find(G, dbgr=dbgr)

    return G


def init_root_node_size_and_harbor(G,
                                   bias=False,
                                   fetch_func=fetch_node):

    # Find the input nodes, list of dicts of G attributes
    input_nodes = fetch_func(type='bias' if bias else 'placeholder', graph=G)

    if len(input_nodes) == 0:
        return G

    # Make sure the functions exist, and that the output sizes specified
    # Then, specify the node state and output sizes
    # Then, add a Dummy Harbor for that input
    # Then, add to node_touch for proper ordering
    for i in input_nodes:
        this_name = i['nickname']
        assert 'functions' in i, 'Input node %s has no functions!' % \
            (this_name)
        assert 'output_size' in i['functions'][0], 'Input node %s has no \
            output size specified!' % (this_name)
        # [batch, height, width, channels], batch is usually set to None
        G.node[this_name]['output_size'] = [BATCH_SIZE] \
            + i['functions'][0]['output_size']
        G.node[this_name]['state_size'] = G.node[this_name]['output_size'][:]
        # Create a Harbor instance
        G.node[this_name]['harbor'] = Harbor(
            incoming_sizes={this_name: (G.node[this_name]['output_size'], [])},
            node_name=this_name
        )
        # G.node[this_name]['harbor'] = Harbor_Dummy(
        #     G.node[this_name]['output_size'],
        #     input_=not bias)

        G.graph['input_touch_order'].append(this_name)

    return G


def root_node_size_find(G, dbgr=dbgr_silent):

    dbgr('Starting the node traversal for size calculation.')

    # Store the correct initialization order for the nodes:
    if 'input_touch_order' not in G.graph:
        G.graph['input_touch_order'] = []
    if 'touch_order' not in G.graph:
        G.graph['touch_order'] = []

    # Initialize inputs
    G = init_root_node_size_and_harbor(G, bias=False)

    # Initialize biases
    G = init_root_node_size_and_harbor(G, bias=True)

    return G


def all_node_size_find(G,
                       root_nodes=None,
                       fetch_func=fetch_node,
                       dbgr=dbgr_silent):

    if not root_nodes:
        root_nodes = [i for i in G.nodes() if len(G.predecessors(i)) == 0]

    while not all(['output_size' in G.node[i] for i in G.nodes()]):
        for node in G.nodes():
            if 'output_size' in G.node[node]:
                # Current node has been calculated already, skip!
                continue
            else:
                if all(['output_size' in G.node[i]
                        for i in get_forward_pred(G, node)]):
                    dbgr('Counting up the sizes for node %s' % (node))
                    # All the predecessors have been traversed, we can now
                    # proceed

                    # First, gather all the necessary info about the
                    # current node:
                    current_info = fetch_func(node, graph=G)[0]

                    # Then, gather the sizes of all incoming nodes into a
                    # dict: {'nickname': (size, all_simple_paths)}
                    incoming_sizes = {}
                    for pred in get_forward_pred(G, node):
                        incoming_sizes[pred] = \
                            (G.node[pred]['output_size'], list(
                                chain(*[nx.all_simple_paths(G, st, pred)
                                        for st in root_nodes])))

                    dbgr('Incoming sizes of %s: %s' % (node, incoming_sizes))

                    # # Create a Policy instance
                    # current_policy = Policy()

                    # Create a Harbor instance
                    G.node[node]['harbor'] = Harbor(incoming_sizes,
                                                    policy=None,
                                                    node_name=node)

                    # Extract the desired_size from Harbor:
                    desired_size = G.node[node]['harbor'].get_desired_size()

                    # Find the size of the state:
                    current_state_size = \
                        utility_functions.chain_size_crunch(
                            desired_size, current_info['functions'])

                    # Find the size of the output (same as state for now):
                    current_out_size = current_state_size[:]

                    # Update node_out_size and node_state_size
                    G.node[node]['output_size'] = current_out_size[:]
                    G.node[node]['state_size'] = current_state_size[:]

                    # Add the current node to the node_touch list for further
                    # Tensor creation order
                    G.graph['touch_order'].append(node)

                    dbgr(' ... done!', newline=False)

                    # break the for loop after modifying the node_out_size dict
                    break
                else:
                    # There are other nodes that need to be calculated, skip
                    continue

    dbgr()
    dbgr('Final Node Output Sizes:', newline=False)
    for i in sorted(G.nodes()):
        dbgr('   ' + i.ljust(max([len(j) for j in G.nodes()]))
             + ':' + str(G.node[i]['output_size']), newline=False)
    dbgr(newline=False)

    dbgr('\nAll sizes calculated!')

    return G


def get_forward_pred(G, node):
    # Get only those predecessors that are not feedback
    # First gather a list of all predecessors
    a = G.predecessors(node)
    # Then filter out the nodes that are feedback
    b = [i for i in a if not G.edge[i][node]['feedback']]
    return b
