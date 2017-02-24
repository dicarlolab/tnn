"""
Unroller!
"""

import networkx as nx
import tensorflow as tf
from utility_functions import get_forward_pred


def unroller_call(input_sequence, G, ntimes=None, last=None):
    # Calculate ntimes from length of input_sequence and nx_graph structure

    # Find the root nodes and the output node
    root_nodes = [i for i in G.nodes() if len(get_forward_pred(G, i)) == 0]
    if not last:
        last_nickname = [n for n in G.nodes() if
                         len([i for i in nx.neighbors(G, n)
                              if not G.edge[n][i]['feedback']]) == 0][0]
        last = G.node[last_nickname]['tf_cell']

    # Then, find the longest path from a root to the output in graph G:
    max_path_length = 0
    for r in root_nodes:
        max_path_length = max(
            max_path_length,
            max([len(p) for p in nx.all_simple_paths(G, r, last_nickname)])
        )

    # If input sequence is a list of values, we take the length of that list
    # to be the unroll number
    if isinstance(input_sequence['images'], list):
        if ntimes:
            raise Exception('Cannot supply list for input_sequence AND '
                            'specify ntimes. Pick one of them.')
        ntimes = len(input_sequence['images'])

        #if len(input_sequence['images']) < max_path_length:
        #    raise Exception('The input sequence is not long enough! '
        #                    '%s is shorter than required %s' %
        #                    (len(input_sequence['images']), max_path_length))
    else:
        if ntimes is None:
            ntimes = max_path_length
        else:
            pass
            #if ntimes < max_path_length:
            #    raise Exception('Specified unroll length is not long enough! '
            #                    '%s is shorter than required %s' %
            #                    (ntimes, max_path_length))

    # Make the code generally assume list structure of input:
    input_is_list = isinstance(input_sequence['images'], list)
    # Loop over time

    print("NTIMES", ntimes)
    for t in range(ntimes):
        # Loop over nodes
        for node in G.nodes():
            # Gather inputs
            preds = G.predecessors(node)
            if len(preds) == 0:
                # input node:
                # This will have input_sequence addressing when
                # I multiply the input_sequence through time - TO DO!
                # For now every input cell gets the same image as the output
                # for every time step
                G.node[node]['tf_cell'].update_outputs(
                    input_sequence['images'][t] if input_is_list else input_sequence['images'])
                G.node[node]['tf_cell'].update_states(
                    input_sequence['images'][t] if input_is_list else input_sequence['images'])
            else:
                inputs = {p: G.node[p]['tf_cell'].get_output(t-1) for p in preds}
                # Compute output and state
                curstate = G.node[node]['tf_cell'].get_state()
                out, state = G.node[node]['tf_cell'](inputs, curstate)
                G.node[node]['tf_cell'].update_outputs(out)
                G.node[node]['tf_cell'].update_states(state)
        tf.get_variable_scope().reuse_variables()

    print 'Unroller successfully unrolled Graph! %s nodes unrolled' % (
        len(G.nodes()) * ntimes)

    return G, last
