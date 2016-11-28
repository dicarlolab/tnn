def _construct_graph(layers, bypasses):
    """
    Constructs networkx DiGraph based on bypass connections
    :param bypasses: list of tuples (from, to)
    :param N_cells: number of layers not including last FC for logits
    :return: Returns a networkx DiGraph where nodes are layer #s, starting
    from 0 (input) to N_cells + 1 (last FC layer for logits)
    """
    graph = nx.DiGraph()
    nlayers = len(layers)
    graph.add_node('0', cell=None, name='input')
    prev_node = '0'
    names = []
    for node, layer in enumerate(layers):
        node = str(node + 1)
        cell = layer()  # initialize cell
        graph.add_node(node, cell=cell, name=cell.scope)
        graph.add_edge(str(int(node)-1), node)

    #  adjacent layers
    # graph.add_edges_from([(names[i], names[j]) for i,j in bypasses])  # add bypass connections
    graph.add_edges_from([(str(i), str(j)) for i,j in bypasses])
    # print(graph.nodes())

    # check that bypasses don't add extraneous nodes
    if len(graph) != nlayers + 1:
        import pdb; pdb.set_trace()
        raise ValueError('bypasses list created extraneous nodes')

    return graph