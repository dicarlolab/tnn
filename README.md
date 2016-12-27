# UniCycle Architecture Builder
Basic overview:

Step 1
=======
 High-level description of the system is fed in via a JSON object-bearing
 file that is passed in as a command line argument when the script is executed

Step 2
=======
 A Network-X model of the system is composed from the JSON file, containing
 all the necessary metadata and progress/bypass/feedback connector information

Step 3
=======
 The NODES of the Network-X model are converted into appropriate General
 Functional Cells, and their internal metadata is used to populate the
 functional space inside of each of the cells

Step 4
=======
 The EDGES of the Network-X model (progress/bypass/feedback) are converted to
 progress/bypass/feedback connections in the TF graph. This is done by adding
 pointers/references from every TF object to every other object that it
 receives inputs from, from both the previous and the current time steps. 

 Proper sizes for the input/state/output sizes of all the TF Cells are
 calculated and accounted for.

Step 5
=======
 Proper RNN unrolling of nodes within 1 time step is performed. This is almost
 cheating, as the RNN is essentially unrolled through a single time step but
 memoized states it's parent and predecessor Cells are queried and accounted
 for, creating the illusion of a true RNN unroll. In reality, DAG forward 
 structure is preserved

Let's kick ass


To be updated soon ...

