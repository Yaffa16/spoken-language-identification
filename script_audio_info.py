"""
This script creates a CSV file containing useful information about the audio
files.

Dependencies:
    - sox (soxi): this module uses the soxi software to get audio information.
    Please be sure you have this software installed in your PATH system
    variable. More details in http://sox.sourceforge.net
"""

import glob
import argparse
import json
from tqdm import tqdm
import subprocess
import sys
import os
from collections import defaultdict


def extract_info(audio_path, command: str):
    """
    Extracts information of an audio file.

    :param audio_path: str
        Path to the audio file.
    :param command: str
        Command to run.
    :return: str
        The stdout resulted from the command.
    """
    process = subprocess.Popen(command + ' "' + audio_path + '" ',
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    return stdout.decode(sys.stdout.encoding).replace('\r', '').split('\n')


def extract_all_info(audio_path):
    """
    Extracts information of an audio file.

    This function runs soxi system command.

    :param audio_path: str
        Path to the audio file.
    :return: dict
        A dictionary with the respective information acquired.
    """
    info = extract_info(audio_path, 'soxi')
    extractions = defaultdict(str)
    for i in info:
        i = i.split(': ')
        if len(i) == 2:
            extractions[i[0].strip()] = i[1].strip()
    return extractions


def append_csv(files_path, output_file, base_name=None,
               append_command: tuple=None):
    """
    Extracts and appends information of the files to the CSV file provided.

    :param files_path: list
        List of path of files to extract information.
    :param output_file: str
        Path to the CSV file.
    :param base_name: str
        Name of the base. Optional.
    :param append_command: tuple
        If a new command is provided, a new column will be appended to the file.
        The first item of the tuple must be the column name, and the second,
        the command to run.
        Example: ('length', 'soxi -D')
    :return:
    """
    print('[INFO] extracting info from audio files')
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as file:
            header = 'Corpus,Input File,Channels,Sample Rate,Precision,'\
                     'Duration,File Size,Bit Rate,Sample Encoding'
            header += str(',' + append_command[0] if append_command is not None
                          else '')
            file.write(header + '\n')
    
    with open(output_file, 'a') as file:    
        for file_path in tqdm(files_path):
            tokens = extract_all_info(file_path)
            for token in tokens:
                tokens[token] = tokens[token].replace(',', ' ')
            if base_name is None:
                base_name = os.path.basename(os.path.dirname(file_path))
            line = base_name \
                + ',' + tokens['Input File']\
                + ',' + tokens['Channels']\
                + ',' + tokens['Sample Rate']\
                + ',' + tokens['Precision']\
                + ',' + tokens['Duration']\
                + ',' + tokens['File Size']\
                + ',' + tokens['Bit Rate']\
                + ',' + tokens['Sample Encoding']\
                + ',' + str(extract_info(file_path, append_command[1])[0] if
                            append_command is not None else '')
            file.write(line + '\n')


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Get information about '
                                                 'audio files (corpus).')
    parser.add_argument('corpus', help='Corpus information (JSON file) or '
                                       'path.')
    parser.add_argument('output', help='Output path for the CSV report file.')
    parser.add_argument('--append_length', help='Appends a duration column. '
                                                'The overall command returns '
                                                'the duration of the audio '
                                                'file, but some extra '
                                                'information is returned. Use '
                                                'this option to append a '
                                                'column with only the length '
                                                'of the audio file.',
                        action='store_true', default=False)
    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.corpus
    output = arguments.output
    append_len = arguments.append_length

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

            # Set base samples amount
            bases_json[base]['samples'] = len(all_files_path)
            print('Total of raw files: %d' % len(all_files_path))
            append_csv(all_files_path, output, str(base),
                       append_command=('length', 'soxi -d') if append_len else
                       None)
    else:
        all_files_path = glob.glob(data_dir + '/**/*.wav', recursive=True)
        print('Total of raw files: %d' % len(all_files_path))
        append_csv(all_files_path, output, base_name=None,
                   append_command=('Length', 'soxi -d') if append_len else
                   None)
