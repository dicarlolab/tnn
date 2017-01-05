"""
construct_G(links, draw=False, dbgr=dbgr_silent)
    This function build the raw networkx graph G

    Takes:
     - Links    <List> List of (from, to) Tuples
     - 'dbgr'   <Function>  Default Debugger function for debugger output
     - 'draw'    <Bool>  Indicate whether to plot the graph or not

    Returns
     - G    <NX_DiGraph>  Network-X DiGraph of G



construct_H(G, root_nodes=None, dbgr=dbgr_silent)
    This function build the raw networkx graph G

    Takes:
     - G    <NX_DiGraph>  Network-X DiGraph of G
     - 'root_nodes'    <List>  List of string nicknames of root nodes
     - 'dbgr'   <Function>  Default Debugger function for debugger output
    Returns
     - H    <NX_DiGraph>  Network-X DiGraph of forward-only H
"""

# System things
import sys
import os
import json
from unicycle_settings import *
from dbgr import dbgr_silent
from itertools import chain

# Import Network-X support libraries
import networkx as nx
import matplotlib.pyplot as plt


def construct_G(links, draw=False, dbgr=dbgr_silent):
    dbgr('Building Network-X Raw DiGraph...')
    # We create a Network-X DiGraph G and populate it using the links.
    G = nx.DiGraph()
    G.add_edges_from(links)
    # G.add_edge('fc_8','conv_3')

    dbgr('Network-X Raw Graph created! Nodes: ', newline=False)
    dbgr('    ' + '\n    '.join(sorted(G.nodes())))

    # Now we find the starting nodes - these will be nodes without
    # predecessors, nodes that are either
    #   a) input nodes with placeholders
    #   b) bias nodes
    # The list of these nodes will be collected in the `root_nodes` variable
    root_nodes = [i for i in G.nodes() if len(G.predecessors(i)) == 0]

    dbgr('Root nodes: ', newline=False)
    dbgr('    ' + '\n    '.join(sorted(root_nodes)))

    if draw:
        # Draw the Graph
        dbgr('Drawing graph')
        pos = nx.circular_layout(G)
        labels = {str(i): str(i).upper() for i in G.nodes()}
        nx.draw(G, pos)
        nx.draw_networkx_labels(G, pos, labels)
        dbgr('Close the Matlab preview window to continue')
        plt.show()
        dbgr()

    return G, root_nodes


def construct_H(G, root_nodes=None, dbgr=dbgr_silent):
    dbgr('Finding all non-ancestral dependeny links in the Graph...', 1)

    # If root_nodes is not supplied, we calculate it from G (again):
    if not root_nodes:
        root_nodes = [i for i in G.nodes() if len(G.predecessors(i)) == 0]

    # Gotta love string comprehensions :)
    # Add only those edges that lead forward (the target is not an ancestor)
    forward = [(fr, to) for (fr, to) in G.edges()
               if not any([to in i for i in \
                           # All paths lead to Rome! (from all roots to `to`)
                           list(chain(*[nx.all_simple_paths(G, first, fr) \
                                for first in root_nodes]))
                           ])
               ]

    # Let's create a forward-only copy of G --> H
    H = nx.DiGraph(forward)

    dbgr('done!')
    dbgr('Links: ', newline=False)
    dbgr('    ' + '\n    '.join([str(i) for i in forward]))

    dbgr('Nodes of forward-only Graph: ', newline=False)
    dbgr('    ' + '\n    '.join(sorted(H.nodes())))

    return H
