"""
This script creates usable data sets for training. All data of the original
bases will be concatenated before processing.

Dependencies:
    - sox: this module uses the sox software to convert audio files to wav
    format. Please be sure you have this software installed in your PATH system
    variable. More details in http://sox.sourceforge.net
"""
import glob
import concurrent.futures
import os
import json
import argparse
import importlib
import numpy as np
import wave
from collections import defaultdict
from tqdm import tqdm


def files_to_process(files_list: list, output_dir: str,
                     contains_extension=True, is_base_name=True) -> list:
    """
    Build a list of remaining files to process.

    This function checks the data set integrity.

    :param files_list: list
        The original list of file names to process.
        Only names is considered by default (is_base_name=True).
    :param output_dir: str
        The path of the data set (output directory).
    :param contains_extension: bool
        Must be True if the file names contains file extensions (for instance
        file.wav). Default to True.
    :param is_base_name: bool
        Consider the provided list of files as a list of names. If you provide
        a list of paths, set this as False. Default to True.

    :return: list
        A list containing the remaining files to process (in the format of the
        provided list).
    """
    # Make a copy of the list
    remaining_files = files_list.copy()
    print('[INFO] checking files in', output_dir)
    counter = 0
    # List the output files in the directory
    output_files = os.listdir(output_dir)
    # Iterate over file names and eliminate from this_files if the respective
    # file already exists.
    for el in files_list:
        if not is_base_name:
            file_name = os.path.basename(el)
        else:
            file_name = el
        if contains_extension:
            file_name = str(file_name.split('.')[-2])
        if file_name + '.wav' in output_files:
            remaining_files.remove(el)
            counter += 1
    print('There are {} files remaining to process, '
          'and {} files in {}.'.format(len(files_list) - counter, counter,
                                       output_dir))
    return remaining_files


def create_dataset(file_list: list, dataset_dir: str, trim_interval: int,
                   normalize=True, num_workers: int = None, name: str=None,
                   exist_ok=False, verbose_level=0):
    """
    Creates a dataset.

    :param file_list: list
        List of files to process (paths).
    :param dataset_dir: str
        Output directory (data set).
    :param name:
        Output name (concatenated audio).
    :param num_workers: int
        The maximum number of processes that can be used to execute the given
        calls.
    :param trim_interval: tuple, shape=(start, end)
        Trim audio. Default to None (no trim will be performed).
    :param normalize: bool
        Normalizes audio.
    :param exist_ok: bool
        If the target name already exists, raise an FileExistsError if exist_ok
        is False. Otherwise the file will be replaced.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """

    os.makedirs(dataset_dir, exist_ok=True)
    print('[INFO] creating data set [%s]' % dataset_dir)

    name = name + '.wav' if name is not None else 'temp.wav'

    # Concat all files
    chunk_size = 40  # Windows string cmd line limitation
    print('[INFO] concatenating chunks of size %s' % chunk_size)
    chunks = [file_list[i:i + chunk_size]
              for i in range(0, len(file_list), chunk_size)]
    for i in range(len(chunks)):
        chunks[i] = ' '.join(chunks[i])

    data_out = dataset_dir + os.sep + name

    if os.path.isfile(data_out) and not exist_ok:
        raise FileExistsError()
    elif os.path.isfile(data_out):
        os.system('sox -V%s %s %s %s' % (verbose_level, chunks[0], data_out,
                                         data_out))
    else:
        os.system('sox -V%s %s %s' % (verbose_level, chunks[0], data_out))

    for i in tqdm(range(1, len(chunks))):
        os.system('sox -V%s %s %s %s' % (verbose_level, chunks[i], data_out,
                                         data_out))

    # Trim audio
    if trim_interval is not None:
        audio = wave.open(data_out, 'r')  # todo check size
        duration = audio.getnframes() / float(audio.getframerate())
        audio.close()
        trims = np.arange(0, duration, trim_interval)
        trims = list(zip(trims[0:len(trims) - 1], trims[1:len(trims)]))

        # Make parallel calls to trim the audio file
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
                as executor:
            futures = [
                executor.submit(fn=trim,
                                audio_path=data_out,
                                output_path=dataset_dir + '_' +
                                            str(t[0]) + '_' + str(t[1]),
                                start=t[0],
                                end=t[1],
                                verbose_level=verbose_level)
                for t in trims]

            kwargs = {
                'total': len(futures),
                'unit': 'files',
                'unit_scale': True,
                'leave': True
            }
            for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
                pass

    if normalize:
        # Make parallel calls to normalize audio files
        file_list = glob.glob(dataset_dir + os.sep + '.wav')
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
                as executor:
            futures = [
                executor.submit(fn=norm,
                                audio_path=file_path,
                                verbose_level=verbose_level)
                for file_path in file_list]

            kwargs = {
                'total': len(futures),
                'unit': 'files',
                'unit_scale': True,
                'leave': True
            }
            for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
                pass


def norm(audio_path: str, verbose_level=0):
    """
    Normalizes an audio file using sox software.

    :param audio_path: str
        Path to the audio file.
    :param verbose_level: int
    Verbosity level. See sox for more information.
    """
    os.system('sox -V%s %s gain −n' % (verbose_level, audio_path))


def trim(audio_path: str, output_path: str, start, end, verbose_level=0):
    """
    Trim audio files using sox software.

    :param audio_path: str
        List of files to process (paths).
    :param output_path: str
        Output directory (data set).
    :param start:
        Start position to trim.
    :param end:
        End position to trim.
    :param verbose_level:
        Verbosity level. See sox for more information.
    """
    os.system('sox -V%s %s %s %s %s' % (verbose_level, audio_path, start, end,
                                        output_path))


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a dataset of audio '
                                                 'files in a proper format.')
    parser.add_argument('source', help='Source directory.')
    parser.add_argument('output', help='Output directory.')
    parser.add_argument('--seconds', help='Length of audio files in seconds.')
    parser.add_argument('--workers', help='Define how many process to run in '
                                          'parallel.', default=4, type=int)
    parser.add_argument('--v', help='Change verbosity level (sox output)',
                        default=0)
    parser.add_argument('--check', help='Check output directories ignoring '
                                        'files already processed.',
                        action='store_true')
    os.makedirs('logs/scripts/', exist_ok=True)

    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.source
    output = arguments.output
    seconds = arguments.seconds
    workers = arguments.workers
    verbose = arguments.v

    # Load json info about bases
    with open('bases_test.json') as base_json:
        bases_json = json.load(base_json)

    # The files will be processed as a new base by their language
    files_list_lang = defaultdict(lambda: [])

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

        # Set the output directory of each base
        out_dir = output + os.sep + bases_json[base]['lang'] + os.sep

        # Check processed files if necessary, removing them
        if arguments.check:
            files_paths = files_to_process(all_files_path, out_dir,
                                           contains_extension=True,
                                           is_base_name=False)
        else:
            files_paths = all_files_path

        # Append file paths to the respective language base
        files_list_lang[bases_json[base]['lang']] += files_paths

    print('TOTAL FILES TO BE PROCESSED: %d' %
          (sum(len(b) for b in files_list_lang.values())))

    # Print summaries:
    pandas_found = importlib.util.find_spec('pandas')
    if pandas_found is not None:
        pandas_lib = importlib.import_module('pandas')
        pd = pandas_lib.pandas

        print('\nORIGINAL BASES\n--------------')
        print(pd.DataFrame(bases_json))

        d = dict()
        d['New base'] = list(files_list_lang.keys())
        d['Total samples'] = list(map(lambda x: len(files_list_lang[x]),
                                      files_list_lang))
        print('\nSUMMARY\n-------')
        print(pd.DataFrame(d))

    # Process files of each language as a new base:
    for base in files_list_lang:
        print('[INFO] processing base in "%s"' % base)

        # Call function and make data set
        create_dataset(dataset_dir=output + os.sep + base,
                       file_list=files_list_lang[base],
                       num_workers=workers,
                       verbose_level=verbose,
                       name=base,
                       exist_ok=True,
                       trim_interval=seconds if seconds is not None else None)
