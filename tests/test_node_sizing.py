"""
This is a test suite for the

node_sizing

module. This file includes one or many Test* class(es), which houses all
the test_*() functions for that particular set of tests.

Let's begin.
"""

import pathmagic
import pytest
import networkx as nx

from node_sizing import all_node_sizes
from node_sizing import init_root_node_size_and_harbor
from node_sizing import root_node_size_find
from node_sizing import all_node_size_find
from node_sizing import get_forward_pred


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


class TestGetForwardPred:
    G = nx.DiGraph()
    # Generate all the nodes and links
    # This simulates the correct operation of construct_G
    G.add_edges_from([
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 5),
        (6, 4)],
        feedback=False)
    G[7][5]['feedback'] = True
    G[6][4]['feedback'] = True

    def test_check_forward_preds_on_forward_edges(self):
        """
        Test to see if the predecessor selection is done properly
        for the forward case
        """
        assert get_forward_pred(self.G, 2) == [1]
        assert get_forward_pred(self.G, 3) == [1]
        assert get_forward_pred(self.G, 4) == [1]

        assert get_forward_pred(self.G, 6) == [5]
        assert get_forward_pred(self.G, 7) == [6]

    def test_check_forward_preds_on_feedback_edges(self):
        """
        Test to see if the predecessor selection is done properly
        for the forward case
        """
        assert get_forward_pred(self.G, 4) == [1]
        assert set(get_forward_pred(self.G, 5)) == set([2, 3, 4])
        assert get_forward_pred(self.G, 4) == [1]

        assert get_forward_pred(self.G, 6) == [5]


# class TestInitRootNodeSizeAndHarbor:
#     G = nx.DiGraph()
#     # Generate all the nodes and links
#     # This simulates the correct operation of construct_G
#     G.add_edges_from([
#         (1, 2),
#         (1, 3),
#         (1, 4),
#         (2, 5),
#         (3, 5),
#         (4, 5),
#         (5, 6),
#         (6, 7),
#         (7, 5),
#         (6, 4)],
#         feedback=False)
#     G[7][5]['feedback'] = True
#     G[6][4]['feedback'] = True

#     def test_check_forward_preds_on_forward_edges(self):
#         """
#         Test to see if the predecessor selection is done properly
#         for the forward case
#         """
#         assert get_forward_pred(self.G, 2) == [1]
#         assert get_forward_pred(self.G, 3) == [1]
#         assert get_forward_pred(self.G, 4) == [1]

#         assert get_forward_pred(self.G, 6) == [5]
#         assert get_forward_pred(self.G, 7) == [6]

#     def test_check_forward_preds_on_feedback_edges(self):
#         """
#         Test to see if the predecessor selection is done properly
#         for the forward case
#         """
#         assert get_forward_pred(self.G, 4) == [1]
#         assert set(get_forward_pred(self.G, 5)) == set([2, 3, 4])
#         assert get_forward_pred(self.G, 4) == [1]

#         assert get_forward_pred(self.G, 6) == [5]
