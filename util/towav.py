"""
towav: convert audio files to wav format

Dependencies:
    - mpg123: this module uses the mpg123 software to convert audio files to wav
    format. Please be sure you have this software installed in your PATH system
    variable. More details in https://www.mpg123.de/index.shtml.
    - sox: this module uses the sox software to convert audio files to wav
    format. Please be sure you have this software installed in your PATH system
    variable. More details in http://sox.sourceforge.net
"""

import os
import argparse
import glob


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
            raise RuntimeError('Not able to convert file', input_path,
                               output_path)


def sox_convert(input_path: str, output_path: str, check=True,
                verbose_level=0):
    """
    Convert an audio file to wav format.

    :param input_path: str
        Input file path
    :param output_path: str
        Input file path
    :param check: bool
        Check if the output file exists
    :param verbose_level: int

    Currently supported formats: formats supported by sox software.
    """
    os.system('sox -V{} {} {}'.format(verbose_level, input_path, output_path))
    if check:
        if not os.path.isfile(output_path):
            raise RuntimeError('Not able to convert file', input_path,
                               output_path)


if __name__ == '__main__':
    from tqdm import tqdm
    import random
    # from shutil import copyfile
    import string
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Convert audio files to wav '
                                                 'format.')
    parser.add_argument('data_dir', help='Source directory of the audio '
                                         'files.')
    parser.add_argument('out_dir', help='Output directory of converted '
                                        'files.')
    parser.add_argument('--check', help='Check if the files are being '
                                        'converted. If not, the program will '
                                        'stop running.',
                        action='store_true', default=False)
    parser.add_argument('-name', help='Choose from the original name or a '
                                      'random name to generate for each '
                                      'converted file.',
                        choices=['original', 'random'])
    parser.add_argument('--verbose_level', help='Verbosity level', default=0,
                        choices=[0, 1, 2], type=int)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = glob.glob(args.data_dir + '/**/*.*', recursive=True)
    for file in tqdm(files):
        name = file.split(os.sep)[-1].split('.')[-2] if args.name == 'original'\
            else ''.join(random.choice(string.ascii_letters + string.digits)
                         for _ in range(12))
        # copyfile(file, args.out_dir + os.sep + name + '.' +
        #          file.split(os.sep)[-1].split('.')[-1])
        sox_convert(file, args.out_dir + os.sep + name + '.wav',
                    check=args.check, verbose_level=args.verbose_level)
