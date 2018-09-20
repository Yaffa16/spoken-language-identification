"""
This script creates a CSV file containing useful information about the audio
files.

Dependencies:
    - sox (soxi): this module uses the sox software to convert audio files to
    wav format. Please be sure you have this software installed in your PATH
    system variable. More details in http://sox.sourceforge.net
"""

import glob
import argparse
import json
import wave
import contextlib
import os
from tqdm import tqdm
import subprocess
import re
import sys
from collections import defaultdict


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Get more information about '
                                                 'bases.')
    parser.add_argument('corpus', help='Corpus information (JSON file)')
    parser.add_argument('output', help='Output path for CSV report file.')

    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.corpus
    output = arguments.output

    # Load json info about bases
    with open(data_dir) as base_json:
        bases_json = json.load(base_json)

    with open(output, 'w') as csv_file:
        csv_file.write('Corpus,Input File,Channels,Sample Rate,Precision,'
                       'Duration,File Size,Bit Rate,Sample Encoding\n')
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

            for file_path in tqdm(all_files_path):
                process = subprocess.Popen(['soxi', file_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT)
                stdout, stderr = process.communicate()
                info = stdout.decode(sys.stdout.encoding)
                info = info.replace('\r', '')
                info = info.split('\n')
                tokens = defaultdict(str)
                for i in info:
                    i = i.split(': ')
                    if len(i) == 2:
                        tokens[i[0].strip()] = i[1].strip()

                csv_file.write(base + ',' + tokens['Input File']
                               + ',' + tokens['Channels']
                               + ',' + tokens['Sample Rate']
                               + ',' + tokens['Precision']
                               + ',' + tokens['Duration']
                               + ',' + tokens['File Size']
                               + ',' + tokens['Bit Rate']
                               + ',' + tokens['Sample Encoding']
                               + '\n')