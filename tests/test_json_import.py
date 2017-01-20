"""
This is a test suite for the

json_import

module. This file includes one or many Test* class(es), which houses all
the test_*() functions for that particular set of tests.

Let's begin.
"""

import pathmagic
import pytest
from json_import import json_import


class TestJSON:
    def test_broken_file_name(self):
        """
        Test to see if broken file name throws up exception
        """
        with pytest.raises(AttributeError):
            n, l = json_import(344)

        with pytest.raises(IOError):
            n, l = json_import('no_extension_provided')

    def test_mixed_up_arguments(self):
        """
        Test to see if incorrectly supplied dbgr throws exception
        """
        with pytest.raises(TypeError):
            n, l = json_import(dbgr='obviously_not_a_function')

    def test_nodes_have_proper_fields(self):
        """
        Test to see if the JSON nodes are properly imported
        """
        n, _ = json_import()
        # Should be a list
        assert isinstance(n, type([])), 'Node list not a list!'
        # Not empty
        assert len(n) > 0, 'Nodes list is empty!'

        for node in n:
            # Nodes are dictionaries
            assert isinstance(node, type({}))
            assert 'type' in node
            assert 'nickname' in node
            assert 'functions' in node

    def test_links_have_proper_fields(self):
        """
        Test to see if the JSON links are properly imported
        """
        _, l = json_import()
        # Should be a list
        assert isinstance(l, type([])), 'Link list not a list!'
        # Not empty
        assert len(l) > 0, 'Links list is empty!'

        for link in l:
            # Links are tuples
            assert isinstance(link, type(('', '')))
            assert len(link) == 2, 'Links are supposed to be 2 long!'
