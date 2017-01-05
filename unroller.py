"""
Unroller!
"""

import networkx as nx


def unroller_call(input_sequence, G, repo, last=None):
    # Calculate ntimes from length of input_sequence and nx_graph structure

    ntimes = len(input_sequence)
    # Loop over time
    for t in range(ntimes):
        # Loop over nodes
        for node in repo:
            # Gather inputs
            parents = G.predecessors(node)
            inputs = {p: repo[p].previous() for p in parents}
            # Compute output and state
            out, state = repo[node](input_sequence[t],
                                    repo[node].get_state())

            repo[node].memoize(out)
            repo[node].update_state(state)

    last = [n for n in repo if
            len([i for i in nx.all_neighbors(G, n)]) == 0] \
        if not last else last

    print last, 'jhbjghdfbkjhgbdsfkjhgbdkjfhdgks'

    return repo, last
