import utility_functions
from GenFuncCell import GenFuncCell


def initialize_nodes(nodes,
                     node_out_size,
                     node_state_size,
                     node_harbors,
                     node_input_touch,
                     node_touch,
                     fetch_func=utility_functions.fetch_node):
    # Repository of all the Tensor outputs for each Node in the TF Graph
    repo = {}

    for i in node_input_touch:
        current_info = fetch_func(i, node_storage=nodes)[0]

        # Initialize the TF Placeholder for this input
        this_input = GenFuncCell(harbor=node_harbors[i],
                                 state_fs=[],
                                 out_fs=[],
                                 state_fs_kwargs=[],
                                 out_fs_kwargs=[],
                                 memory_kwargs={},
                                 output_size=node_out_size[i],
                                 state_size=node_state_size[i],
                                 scope=str(i))

        repo[i] = this_input

    # Now, let's initialize all the nodes one-by-one
    for node in node_touch:
        current_info = fetch_func(node, node_storage=nodes)[0]

        sfk = utility_functions.assemble_function_kwargs(
            current_info['functions'],
            node_harbors[node].desired_size,
            node)

        # Let's initiate TF Node:
        tf_node = GenFuncCell(harbor=node_harbors[node],
                              state_fs=[str(f['type'])
                                        for f in current_info['functions']],
                              out_fs=[],
                              state_fs_kwargs=sfk,
                              out_fs_kwargs=[],
                              memory_kwargs={},
                              output_size=node_out_size[node],
                              state_size=node_state_size[node],
                              scope=str(node))

        repo[node] = tf_node

    return repo
