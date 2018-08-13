"""
towav: convert audio files to wav format

Dependencies:
- mpg123: this module uses the mpg123 software to convert audio files to wav
format. Please be sure you have this software installed in your PATH system
variable. More details in https://www.mpg123.de/index.shtml.
- sox: this module uses the mpg123 software to convert audio files to wav
format. Please be sure you have this software installed in your PATH system
variable. More details in
"""

import os
import argparse


def mpg_convert(input_path: str, output_path: str, check=True):
    """
    Convert an audio file to wav format.

    :param input_path: str
        Input file path
    :param output_path: str
        Input file path
    :param check: bool
        Check if the output file exists

    Currently supported formats: formats supported by mpg123 software.
    """
    os.system('mpg123 -w ' + output_path + ' ' + input_path)
    if check:
        if not os.path.isfile(output_path):
            raise RuntimeError('Not able to convert file')


def sox_convert(input_path: str, output_path: str, check=True):
    """
    Convert an audio file to wav format.

    :param input_path: str
        Input file path
    :param output_path: str
        Input file path
    :param check: bool
        Check if the output file exists

    Currently supported formats: formats supported by sox software.
    """
    os.system('sox ' + input_path + ' ' + output_path)
    if check:
        if not os.path.isfile(output_path):
            raise RuntimeError('Not able to convert file')


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Convert audio files to wav '
                                                 'format.')
    parser.add_argument('data_dir', help='Source directory of the audio '
                                         'files.')
    parser.add_argument('out_dir', help='Output directory of converted '
                                        'files.')
    args = parser.parse_args()

    file_names = os.listdir(args.data_dir)
    for i, line in enumerate(file_names):
        mpg_convert(args.data_dir + '/' + line,
                    args.out_dir + "/" + str(line[:-4]) + '.wav')
        print("Processed {} of {} files".format(i, len(file_names)))
