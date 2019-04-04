"""
This script executes hyper parameters searches
"""


if __name__ == '__main__':
    import argparse
    import json
    import os
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Hyper parameters automatic '
                                                 'search')
    parser.add_argument('data', help='CSV file containing paths and labels for'
                                     'training.')
    parser.add_argument('parameters', help='A JSON file with the hyper '
                                           'parameters space to search.')
    parser.add_argument('--experiments', help='Output directory for '
                                              'experiments.')
    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.data
    output_dir = arguments.output
    parameters_file = arguments.parameters

    if os.path.isfile(parameters_file):
        with open(parameters_file) as json_file:
            parameters_space = json.load(json_file)
