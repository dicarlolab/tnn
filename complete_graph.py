def _complete_graph(base_graph, links):
    """
    inputs:
        base_graph = networkx object specifying the convnet
        links = list of pairs of layers to be connected, possibly with a specification 
                of what type of connection should be made (e.g. (un)pooling + concatenation, 
                etc)    
    outputs:
        networkx object whose nodes correspond to convnet layers and relevant connecting 
        operations and whose edges contain the original convnet information flow as
        well as the ones described by the "links" input

        Question:  should we put the rnn cell creation code here, so that each of the nodes
        of the output graph has a "cell" attribute, containing the rnn cell?  Or should
        it just contain the information so that the graph_rnn funciton below can create
        the actual rnn cell operation?  Don't know.
    """



    #do creation of poolings/unpoolings/concatenation/etc  as currently exemplified
    #in lines 331-357 of models.py.   Presumably we might want to generalize what is 
    #currently there to handle other types of connectors (we can talk about this realtime). 

    # for relevant time points: gather inputs, pool, and concatenate
    for t in range(layer['first'], layer['last'] + 1):
        # concatenate inputs (pooled to right spatial size) at time t
        # incoming_shape = layer_sizes[j - 1]['output']  # with no bypass
        # if node == 5 and t == 2: import pdb; pdb.set_trace()
        if len(layer['cell'].state_size) == 4:
            if n == 1:
                output_size = graph.node['0']['outputs'][0].get_shape().as_list()[1]
            else:
                output_size = graph.node[str(n-1)]['cell'].output_size[1]

            inputs_t = []
            for parent in sorted(parents):
                input_tp = _maxpool(
                    input_=graph.node[parent]['outputs'][t - 1],
                    out_spatial=output_size,
                    kernel_size=bypass_pool_kernel_size,
                    name='bypass_pool')
                inputs_t.append(input_tp)

            # concat in channel dim
            # import pdb; pdb.set_trace()
            layer['inputs'].append(tf.concat(3, inputs_t))

        else:  # if input is 2D (ex: after FC layer)
            # print('FC')
            # no bypass to FC layers beyond first FC
            if len(parents) != 1:
                raise ValueError('No bypass to FC layers '
                                 'beyond first FC allowed')
            inputs_t = graph.node[parents[0]]['outputs'][t - 1]
            layer['inputs'].append(inputs_t)
        # print('inputs at t = {}: {}'.format(t, inputs_t))