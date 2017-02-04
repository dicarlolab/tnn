"""
This is a test suite for the

basic_training

module. This file includes one or many Test* class(es), which houses all
the test_*() functions for that particular set of tests.

Let's begin.
"""
import pathmagic
import train_mnist


class TestBasicTraining:
    def test_train_simple_mnist(self):
        print 'This test requires that the MongoDB database is up and running!'
        assert train_mnist.main(100)
