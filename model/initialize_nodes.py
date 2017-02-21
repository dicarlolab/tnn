from utility_functions import fetch_node, assemble_function_kwargs
from GenFuncCell import GenFuncCell


def initialize_nodes(G, fetch_func=fetch_node, train=False):

    for i in G.graph['input_touch_order']:
        current_info = fetch_func(i, graph=G)[0]

        # Initialize the TF Placeholder for this input
        this_input = GenFuncCell(harbor=G.node[i]['harbor'],
                                 state_fs=[],
                                 out_fs=[],
                                 state_fs_kwargs=[],
                                 out_fs_kwargs=[],
                                 memory_kwargs={},
                                 output_size=G.node[i]['output_size'],
                                 state_size=G.node[i]['state_size'],
                                 scope=str(i))

        G.node[i]['tf_cell'] = this_input

    # Now, let's initialize all the nodes one-by-one
    for node in G.graph['touch_order']:
        current_info = fetch_func(node, graph=G)[0]

        sfk = assemble_function_kwargs(current_info['functions'],
                                       G.node[node]['harbor'].desired_size,
                                       node,
                                       train=train)

        # Let's initiate TF Node:
        tf_node = GenFuncCell(harbor=G.node[node]['harbor'],
                              state_fs=[str(f['type'])
                                        for f in current_info['functions']],
                              out_fs=[],
                              state_fs_kwargs=sfk,
                              out_fs_kwargs=[],
                              memory_kwargs=current_info.get('memory', {}),
                              output_size=G.node[node]['output_size'],
                              state_size=G.node[node]['state_size'],
                              scope=str(node))

        G.node[node]['tf_cell'] = tf_node

    return G
