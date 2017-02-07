"""
Unroller!
"""

import networkx as nx
import tensorflow as tf


def unroller_call(input_sequence, G, last=None):
    # Calculate ntimes from length of input_sequence and nx_graph structure

    ntimes = 10     # This is a hack, should be len(input_sequence)
    # Loop over time
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
                G.node[node]['tf_cell'].update_outputs(input_sequence['images'])
                G.node[node]['tf_cell'].update_outputs(input_sequence['images'])
            else:
                inputs = {p: G.node[p]['tf_cell'].get_output(1) for p in preds}
                # Compute output and state
                curstate = G.node[node]['tf_cell'].get_state()
                out, state = G.node[node]['tf_cell'](inputs, curstate)

                G.node[node]['tf_cell'].update_outputs(out)
                G.node[node]['tf_cell'].update_states(state)
        tf.get_variable_scope().reuse_variables()

    last = G.node[
        [n for n in G.nodes() if
         len([i for i in nx.all_neighbors(G, n)]) == 0][0]]['tf_cell'] \
        if not last else last

    print 'Unroller successfully unrolled Graph! %s nodes unrolled' % (
        len(G.nodes()) * ntimes)

    return G, last
