# UniCycle Architecture Builder
Universal Neural Interpretation and Cyclicity Engine (UNICYCLE)

The way this works is the following:

Step 1
=======
 High-level description of the system is fed in via a JSON object-bearing
 file that is passed in as a command line argument when the script is run.
 Here we get the raw metadata from the JSON file and store it in a list of 
 dictionaries, along with all the information about the node

Step 2
=======
 No we create an Network-X graph G for planning purposes. Store the nickname 
 only as the pointer to the node to be instantiated, and then use this 
 nickname to look up relevant node's metadata in the node dictionary list we 
 acquired in step one.

Step 3
=======
 Using the Network-X graph G we will find the longest simple path from start 
 to finish. For now we will use the notation 
    list(nx.bfs_edges(G,first))
 Returns a list of edges.

Step 4
=======
 Using the Network-X graph G we create a parallel graph H and copy the main "spine" into it. 
    [(fr,to) for (fr,to) in G.edges() if not any([to in i for i in nx.all_simple_paths(G,first,fr)])]
 Then, iterate 
 through the "spine" nodes and look up their connections in graph G.

 For each node on the "spine", if the incoming link is from an ancestor then 
 we add it as-is. If, however, the incoming link is not from an ancestor (i.e. 
 incoming from the future), add the link to the node with a ~ attribute in the
 metadata of the node. This is done to let the system know down the line that 
 the past state needs to be accessed. 
 
Step 5
=======
 Once all the connections are made, we start the size calculation. This 
 involves the Harbor of every one of the nodes (here the actual Tensors will 
 be scaled or added or concatenated and the resulting Tensor will be used as 
 input to the functional "conveyor belt"). While the Harbor is the place where 
 the actual resizing happens, we also have the Harbor-Master policy. This 
 policy can be specified in the node metadata in the JSON, or if it isn't 
 specified it can be inferred from the default settings (default subject to 
 modification too). 

 For every NODE:
 - Collect all non-feedback inputs, find their sizes, push list of sizes along 
 with Harbor-Master policy into general Harbor-Master utility function to find 
 ultimate size. 
 - Create a reference dictionary for node metadata that has incoming inputs as 
 keys and scaling values as values. 
 - Calculate all final sizes for all nodes, use for feedback up and down the 
 line.

Step 6
=======
 Tensor creation.

Step 7
=======
 Perform proper RNN unrolling of nodes within 1 time step. This is almost
 cheating, as the RNN is essentially unrolled through a single time step but
 memoized states it's parent and predecessor Cells are queried and accounted
 for, creating the illusion of a true RNN unroll. In reality, DAG forward 
 structure is preserved

Let's kick ass


-------

TO-DO

Things to do by next iteration of Unicycle:

- Pass around NetworkX instead of lists
	v Merge H into G
	v Merge nodes into NX
	- Merge node_out_size into NX
	- Merge node_state_size into NX
	- Merge node_harbors into NX
	- Merge node_input_touch into NX
	- Merge node_touch into NX
	- Copy for H

- Modify GenFuncCell to have methods to memoize outputs and states. Look into pruning.

- Look into Harbor generalization, merge Harbor_Dummy into Harbor

- flake8 things

- Add tests!

- Do training ooooh yeah - timing test and regression test. Benchmarks

