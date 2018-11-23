"""
This script creates a small dataset selecting random instances from another
dataset.
"""
import random
import os
import glob
import string
from shutil import copyfile


def create_dataset(n_instances: int, output_dir: str, files_paths: list,
                   shuffle=True, random_name=False):
    """
    Creates a small dataset from another dataset.

    :param n_instances: int
        Number of instances to select.
    :param output_dir: str
        Output directory.
    :param files_paths: list
        List of audio files paths.
    :param shuffle: bool
        If True, the list of files will be shuffled.
    :param random_name: bool
        If True, a random ASCII string will be generated as the name of the
        output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if shuffle:
        random.shuffle(files_paths)

    for i, file in enumerate(files_paths):
        if i % 100 == 0:
            print('{} done of {}'.format(i, n_instances))
        if i == n_instances:
            break
        n = ''.join(random.choice(string.ascii_letters + string.digits)
                    for _ in range(12)) if random_name \
            else file.split(os.sep)[-1].split('.')[-2]
        copyfile(file, output_dir + os.sep + n + '.' + file.split(os.sep)[-1].
                 split('.')[-1])


if __name__ == '__main__':
    import argparse
    import json
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a small dataset '
                                                 'selecting random instances '
                                                 'from another dataset.')
    parser.add_argument('corpus', help='Corpus information (JSON file) or '
                                       'path.')
    parser.add_argument('output', help='Output path.')
    parser.add_argument('--instances', help='Number of instances to select.',
                        type=int, default=1000)
    parser.add_argument('--shuffle', help='Generates the new dataset randomly.',
                        default=True, action='store_true')
    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.corpus
    output = arguments.output
    rand = arguments.shuffle
    instances = arguments.instances

    # Load json info about bases
    if os.path.isfile(data_dir):
        with open(data_dir) as base_json:
            bases_json = json.load(base_json)
    else:
        bases_json = None

    if bases_json is not None:
        # Get a list of files in each language
        for base in bases_json:
            print('\n[INFO] getting a list of files of base "%s"' % base)

            # Get a list of all files (paths) to process
            all_files_path = glob.glob(bases_json[base]['path'] + '/**/*.' +
                                       bases_json[base]['format'],
                                       recursive=True)
            create_dataset(n_instances=instances, output_dir=output + os.sep +
                           str(base), files_paths=all_files_path, shuffle=rand)
    else:
        all_files_path = glob.glob(data_dir + '/**/*.wav', recursive=True)
        print('Total of raw files: %d' % len(all_files_path))
        create_dataset(n_instances=instances, output_dir=output,
                       files_paths=all_files_path, shuffle=rand)
