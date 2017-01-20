"""
This is a test suite for the

construct_networkx

module. This file includes one or many Test* class(es), which houses all
the test_*() functions for that particular set of tests.

Let's begin.
"""

import pathmagic
import networkx as nx
from construct_networkx import construct_G


class TestConstructG:
    alexnet_nodes = [
        {u'type': u'placeholder',
         u'functions': [
             {u'type': u'placeholder',
              u'output_size': [224, 224, 3]}
         ],
         u'nickname': u'image_input_1',
         u'batch_size': 256},

        {u'type': u'cell',
         u'functions': [
             {u'padding': u'same',
              u'filter_size': [11, 11],
              u'stride': 4,
              u'type': u'conv',
              u'num_filters': 48},
             {u'type': u'relu'},
             {u'depth_radius': 4,
              u'alpha': 0.0001111,
              u'bias': 1,
              u'type': u'norm'},
             {u'padding': u'valid',
              u'stride': 2,
              u'type': u'maxpool',
              u'k_size': 3}],
         u'nickname': u'conv_1'},

        {u'type': u'cell',
         u'functions': [
             {u'padding': u'same',
              u'filter_size': [5, 5],
              u'stride': 1,
              u'type': u'conv',
              u'num_filters': 128},
             {u'type': u'relu'},
             {u'depth_radius': 4,
              u'alpha': 0.0001111,
              u'bias': 1,
              u'type': u'norm'},
             {u'padding': u'valid',
              u'stride': 2,
              u'type': u'maxpool',
              u'k_size': 3}],
         u'nickname': u'conv_2'},

        {u'type': u'cell',
         u'functions': [
             {u'padding': u'same',
              u'filter_size': [3, 3],
              u'stride': 1,
              u'type': u'conv',
              u'num_filters': 192},
             {u'type': u'relu'}],
         u'nickname': u'conv_3'},

        {u'type': u'cell',
         u'functions': [
             {u'padding': u'same',
              u'filter_size': [3, 3],
              u'stride': 1,
              u'type': u'conv',
              u'num_filters': 192},
             {u'type': u'relu'}],
         u'nickname': u'conv_4'},

        {u'type': u'cell',
         u'functions': [
             {u'padding': u'same',
              u'filter_size': [3, 3],
              u'stride': 1,
              u'type': u'conv',
              u'num_filters': 128},
             {u'type': u'relu'},
             {u'padding': u'valid',
              u'stride': 2,
              u'type': u'maxpool',
              u'k_size': 3}],
         u'nickname': u'conv_5'},

        {u'type': u'cell',
         u'functions': [
             {u'type': u'fc',
              u'output_size': 2048},
             {u'type': u'relu'}],
         u'nickname': u'fc_6'},

        {u'type': u'cell',
         u'functions': [
             {u'type': u'fc',
              u'output_size': 2048},
             {u'type': u'relu'}],
         u'nickname': u'fc_7'},

        {u'type': u'cell',
         u'functions': [
             {u'type': u'fc',
              u'output_size': 1000}],
         u'nickname': u'fc_8'}]

    alexnet_links = [
        ('image_input_1', 'conv_1'),
        ('conv_1', 'conv_2'),
        ('conv_2', 'conv_3'),
        ('conv_3', 'conv_4'),
        ('conv_4', 'conv_5'),
        ('conv_5', 'fc_6'),
        ('fc_6', 'fc_7'),
        ('fc_7', 'fc_8')
    ]

    G = construct_G(alexnet_nodes, alexnet_links)

    def test_G_is_networkx(self):
        """
        Test to see if G is a proper NetworkX graph
        """
        assert isinstance(self.G, type(nx.DiGraph())), 'G should be a DiGraph!'

    def test_G_has_enough_nodes(self):
        """
        Test to see if G has all the nodes
        """
        assert len(self.G.nodes()) == \
            len(self.alexnet_nodes), 'G is missing nodes!'

    def test_G_has_enough_links(self):
        """
        Test to see if G has all the links
        """
        assert len(self.G.edges()) == \
            len(self.alexnet_links), 'G is missing edges!'

    def test_G_has_at_least_one_input_cell(self):
        """
        Test to see if G has at least one input (placeholder) cell
        """
        assert any([self.G.node[i]['type'] == 'placeholder'
                    for i in self.G.nodes()])

    def test_all_edges_have_feedback_parameter(self):
        """
        Test to see if all edges in G have a 'feedback' parameter
        """
        assert all(['feedback' in self.G.edge[i[0]][i[1]]
                    for i in self.G.edges()])
