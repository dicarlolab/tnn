"""
This function imports the node information from the JSON file and outputs a
list of dictionaries containing all the node and link information.

Takes:
    --nothing--

Returns
 - Nodes    <List>  List of Dictionaries for every Cell
 - Links    <List>  List of Tuples for every link in the Graph
"""

# import sys
import os
import json
from dbgr import dbgr_silent
import argparse


def json_import(filename=None, dbgr=dbgr_silent):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='sample_mnist.json', type=str)
    args = vars(parser.parse_args())

    if not filename:
        if 'file' not in args:
            dbgr('No JSON settings file given! Scanning current directory...')
            for f in os.listdir('json'):
                if f.endswith('.json'):
                    dbgr('Using first discovered JSON file in current path: %s'
                         % (f.upper()))
                    json_file_name = f
                    break
        else:
            # Fetch the name of the JSON file from the command line args
            json_file_name = args['file']
    else:
        json_file_name = filename

    dbgr('Using JSON file %s for import...' % (json_file_name.upper()), 1)

    with open('json/' + json_file_name) as data_file:
        json_data = json.load(data_file)

    dbgr('done!')

    dbgr('Checking the integrity of the JSON file...', 1)

    assert 'nodes' in json_data, "NODES field not in JSON file!"
    assert len(json_data['nodes']) > 0, "No NODES in the JSON file!"
    assert 'links' in json_data, "LINKS link field not in JSON file!"
    if len(json_data['links']) == 0:
        dbgr('Warning: LINKS empty!')

    nodes = json_data['nodes']
    links = json_data['links']

    links_tuples = [(str(i['from']), str(i['to'])) for i in links]

    dbgr('done!')

    return nodes, links_tuples
