from dbgr import dbgr_silent
from unicycle_settings import *
from utility_functions import fetch_node
from harbor import Harbor, Harbor_Dummy, Policy
from itertools import chain


def all_node_sizes(G, H, nodes, dbgr=dbgr_silent):

    node_out, node_state, node_harbors, node_input_touch, node_touch = \
        root_node_size_find(H=H, nodes=nodes, dbgr=dbgr)

    node_out, node_state, node_harbors, node_touch = \
        all_node_size_find(G=G,
                           H=H,
                           nodes=nodes,
                           node_out_size=node_out,
                           node_state_size=node_state,
                           node_touch=node_touch,
                           dbgr=dbgr)

    return node_out, node_state, node_harbors, node_input_touch, node_touch


def init_root_node_size_and_harbor(nodes,
                                   node_out_size,
                                   node_state_size,
                                   node_harbors,
                                   node_touch_list,
                                   bias=False,
                                   fetch_func=fetch_node):

    # We assume that input image size has been specified
    input_nodes = fetch_func(type='bias' if bias else 'placeholder',
                             node_storage=nodes)
    if not bias:
        # Make sure all the batch sizes are the same:
        assert len(set([i['batch_size'] for i in input_nodes])) == 1, \
            'Batches differ!'
    BATCH_SIZE = input_nodes[0]['batch_size']

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
        node_out_size[this_name] = [BATCH_SIZE] \
            + i['functions'][0]['output_size']
        node_state_size[this_name] = node_out_size[this_name][:]
        node_harbors[this_name] = Harbor_Dummy(node_out_size[this_name],
                                               input_=not bias)
        node_touch_list.append(this_name)

    return node_out_size, node_state_size, node_harbors, node_touch_list


def root_node_size_find(H,
                        nodes,
                        node_out_size=None,
                        node_state_size=None,
                        node_harbors=None,
                        node_input_touch=None,
                        node_touch=None,
                        dbgr=dbgr_silent):

    if not node_out_size:
        node_out_size = {i: None for i in H.nodes()}
    if not node_state_size:
        node_state_size = {i: None for i in H.nodes()}

    dbgr('Starting the node traversal for size calculation. If this step \
        hangscheck the node_out_size validation loop in <Step 5>.')

    # This is a hash table that looks at whether a particular node has been
    # traversed or not (if it's size has been calculated)
    if not node_harbors:
        node_harbors = {}

    # Store the correct initialization order for the nodes:
    if not node_input_touch:
        node_input_touch = []
    if not node_touch:
        node_touch = []

    # Initialize inputs
    node_out_size, node_state_size, node_harbors, node_input_touch = \
        init_root_node_size_and_harbor(nodes=nodes,
                                       node_out_size=node_out_size,
                                       node_state_size=node_state_size,
                                       node_harbors=node_harbors,
                                       node_touch_list=node_input_touch,
                                       bias=False)

    # Initialize biases
    node_out_size, node_state_size, node_harbors, node_touch = \
        init_root_node_size_and_harbor(nodes=nodes,
                                       node_out_size=node_out_size,
                                       node_state_size=node_state_size,
                                       node_harbors=node_harbors,
                                       node_touch_list=node_touch,
                                       bias=True)

    dbgr('Starting Node Output Sizes:', newline=False)
    for i in sorted(node_out_size):
        dbgr('   ' + i.ljust(max([len(j) for j in sorted(node_out_size)]))
             + ':' + str(node_out_size[i]), newline=False)
    dbgr(newline=False)

    return \
        node_out_size, \
        node_state_size, \
        node_harbors, \
        node_input_touch, \
        node_touch


def all_node_size_find(G,
                       H,
                       nodes,
                       node_out_size,
                       node_state_size,
                       node_touch,
                       root_nodes=None,
                       fetch_func=fetch_node,
                       dbgr=dbgr_silent):

    if not root_nodes:
        root_nodes = [i for i in G.nodes() if len(G.predecessors(i)) == 0]

    while not all(node_out_size.values()):
        for node in node_out_size:
            if node_out_size[node]:
                # Current node has been calculated already, skip!
                continue
            else:
                if all([node_out_size[i] for i in H.predecessors(node)]):
                    dbgr('Counting up the sizes for node %s' % (node), 1)
                    # All the predecessors have been traversed, we can now
                    # proceed

                    # First, gather all the necessary info about the
                    # current node:
                    current_info = fetch_func(node, node_storage=nodes)[0]

                    # Then, gather the sizes of all incoming nodes into a
                    # dict: {'nickname':(size,all_simple_paths)}
                    incoming_sizes = {}
                    for pred in G.predecessors(node):
                        incoming_sizes[pred] = \
                            (node_out_size[pred], list(
                                chain(*[nx.all_simple_paths(H, st, pred)
                                        for st in root_nodes])))

                    # Create a Policy instance
                    current_policy = Policy()

                    # Create a Harbor instance
                    node_harbors[node] = Harbor(incoming_sizes,
                                                policy=current_policy,
                                                node_name=node)

                    # Extract the desired_size from Harbor:
                    desired_size = node_harbors[node].get_desired_size()

                    # Find the size of the state:
                    current_state_size = \
                        utility_functions.chain_size_crunch(
                            desired_size, current_info['functions'])

                    # Find the size of the output (same as state for now):
                    current_out_size = current_state_size[:]

                    # Update node_out_size and node_state_size
                    node_out_size[node] = current_out_size[:]
                    node_state_size[node] = current_state_size[:]

                    # Add the current node to the node_touch list for further
                    # Tensor creation order
                    node_touch.append(node)

                    dbgr(' ... done!', newline=False)

                    # break the for loop after modifying the node_out_size dict
                    break
                else:
                    # There are other nodes that need to be calculated, skip
                    continue

    dbgr()
    dbgr('Final Node Output Sizes:', newline=False)
    for i in sorted(node_out_size):
        dbgr('   ' + i.ljust(max([len(j) for j in sorted(node_out_size)]))
             + ':' + str(node_out_size[i]), newline=False)
    dbgr(newline=False)

    dbgr('\nAll sizes calculated!')

    return node_out_size, node_state_size, node_harbors, node_touch
