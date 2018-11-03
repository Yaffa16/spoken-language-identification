"""
This script creates a csv file containing data paths and labels to process.
"""
import argparse
import glob

if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a CSV file '
                                                 'containing data paths and '
                                                 'labels to process.')
    parser.add_argument('pl', help='Paths and labels. First provide the path '
                                   'where the data is, and after each path '
                                   'provide the respective label.'
                                   'For instance, path/to/cats/images cats '
                                   'path/to/dogs/images dogs ...',
                        nargs='+')
    parser.add_argument('f', help='Output path to the CSV file.')
    parser.add_argument('--format', help='Format of the data.',
                        default='png')
    parser.add_argument('--r', help='Get paths recursively.',
                        action='store_true', default=False)

    args = parser.parse_args()

    paths_and_labels = args.pl
    if len(paths_and_labels) % 2 != 0:
        exit('Incorrect number of paths and labels.')
    csv_path = args.f
    recursive = args.r
    data_format = args.format

    paths = paths_and_labels[0::2]
    labels = paths_and_labels[1::2]

    with open(csv_path, 'w') as csv_file:
        for path, label in zip(paths, labels):
            if recursive:
                file_paths = glob.glob(path + '/**/*.' + data_format,
                                       recursive=True)
            else:
                file_paths = glob.glob(path + '/*.' + data_format)
            if len(file_paths) == 0:
                print('Not found. Check the path and format provided. '
                      'Arguments provided:', args)
            for file_path in file_paths:
                csv_file.write(file_path + ',' + label + '\n')
