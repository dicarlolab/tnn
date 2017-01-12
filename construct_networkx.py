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


def construct_G(nodes, links, draw=False, dbgr=dbgr_silent):
    dbgr('Building Network-X Raw DiGraph...')
    # We create a Network-X DiGraph G and populate it using the links.
    #
    # construct_G function returns a Graph that has both forward and backward
    # edges. In this step, we also add an attribute 'feedback' to all edges to
    # prepare a place for specifying that a particular edge is feedback later
    # in the code (we will iterate through the edges and mark edges that are)
    # feedback with a True value for 'feedback' attribute.
    G = nx.DiGraph()
    G.add_edges_from(links, feedback=False)

    # Fill in the node information as attributes for every node:
    for G_node in G.nodes():
        for J_node in nodes:
            if J_node['nickname'] == G_node:
                for att in J_node:
                    G.node[G_node][att] = J_node[att]
                break
            else:
                continue

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

    dbgr('Finding all non-ancestral dependeny links in the Graph...', 1)

    for (fr, to) in G.edges():
        # For every edge in the graph G, we iterate through all the paths
        # that lead to TARGET from SOURCE and see if the edge is a feedback
        # link
        if any([to in i for i in
                list(chain(*[nx.all_simple_paths(G, first, fr)
                             for first in root_nodes]))
                ]):
            # If it is a feedback edge, we change its 'feedback' attribute
            # from False to True
            G.edge[fr][to]['feedback'] = True

    # Now the Graph G has edges that are properly labelled as feedback or not

    dbgr('done!')

    return G
