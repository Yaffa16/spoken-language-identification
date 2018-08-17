"""
This script creates usable data sets for training.

Dependencies:
    - sox: this module uses the sox software to convert audio files to wav
    format. Please be sure you have this software installed in your PATH system
    variable. More details in http://sox.sourceforge.net
"""
import glob
import concurrent.futures
import os
import time
import json
import argparse
import importlib
from collections import defaultdict
from tqdm import tqdm
from shutil import copyfile


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


def create_dataset(dataset_dir: str, file_list: list, num_workers: int=None,
                   pre_processing: callable=None, **kwargs):
    """
    Creates a dataset.

    :param dataset_dir: str
        Output directory (data set).
    :param file_list: list
        List of files to process (paths).
    :param num_workers: int
        The maximum number of processes that can be used to execute the given
        calls
    :param pre_processing: callable
        Pre process data before saving. Default to None, no pre processing will
        be performed.
    :param kwargs:
        Additional kwargs are passed on to the pre processing function.
    """

    os.makedirs(dataset_dir, exist_ok=True)
    print('[INFO] creating data set [%s]' % dataset_dir)

    # Set handler function to process each file
    handler = pre_processing if pre_processing is not None else copyfile

    # Process data in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as \
            executor:
        futures = [executor.submit(handler, file_path, dataset_dir, **kwargs)
                   for file_path in file_list]

        kwargs = {
            'total': len(futures),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
            pass
        with open('logs/scripts/script_create_dataset.txt', 'a') as log:
            log.write('\nExceptions for {} call at {}'.format(handler.__name__,
                                                              time.time()))
            for f in futures:
                if f.exception() is not None:
                    log.write('\n' + str(f.exception()) +
                              ' \n\twhen processing ' + f)


def pre_process(file_path: str, output_dir: str, name: str=None,
                trim_interval: tuple=None, normalize=True,
                verbose_level=0):
    """
    Pre process a file. Use this function to handle raw data.

    :param file_path: str
        Path of the file to process.
    :param output_dir: str
        Path to save the processed file.
    :param name:
        Output name. Default to None (preserves the original name).
    :param trim_interval: tuple, shape=(start, end)
        Trim audio. Default to None (no trim will be performed).
    :param normalize: bool
        Normalizes audio.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    cmd = 'sox -V' + str(verbose_level) + ' ' + file_path
    if name is not None:
        cmd += ' ' + output_dir + os.sep + name + '.wav '
    else:
        cmd += ' ' + output_dir + os.sep + \
               file_path.split(os.sep)[-1].split('.')[-2] + '.wav'
    if trim_interval is not None:
        cmd += ' trim ' + str(trim_interval[0]) + ' ' + str(trim_interval[1])
    if normalize:
        cmd += ' gain −n'
    # print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a dataset of audio '
                                                 'files in a proper format.')
    parser.add_argument('source', help='Source directory.')
    parser.add_argument('output', help='Output directory.')
    parser.add_argument('--seconds', help='Length of audio files in seconds.')
    parser.add_argument('--workers', help='Define how many process to run in '
                                          'parallel.', default=4, type=int)
    parser.add_argument('--check', help='Check output directories ignoring '
                                        'files already processed.',
                        action='store_true')
    parser.add_argument('--check_after', help='Check output directories '
                                              'after processing files and save '
                                              'a list of failed files.',
                        action='store_true')
    parser.add_argument('--v', help='Change verbosity level (sox output)',
                        default=0)
    os.makedirs('logs/scripts/', exist_ok=True)

    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.source
    output = arguments.output
    seconds = arguments.seconds
    workers = arguments.workers
    verbose = arguments.v

    # Load json info about bases
    with open('bases.json') as base_json:
        bases_json = json.load(base_json)

    # The files will be processed as a new base by their language
    files_list_lang = defaultdict(lambda: [])

    # Get a list of files in each language
    for base in bases_json:
        print('\n[INFO] getting a list of files of base "%s"' % base)

        # Get a list of all files (paths) to process
        all_files_path = glob.glob(bases_json[base]['path'] + '/**/*.' +
                                   bases_json[base]['format'], recursive=True)

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
        d['Total samples'] = list(map(lambda x:
                                      len(files_list_lang[x]),
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
                       pre_processing=pre_process,
                       verbose_level=verbose,
                       trim_interval=(0, seconds) if seconds is not None else
                       None)

        # Check base
        if arguments.check_after:
            print("Checking, this might take a while... ")
            with open('logs/scripts/dataset_remaining_files.csv', 'a') as file:
                file.write(base + ',' + 'path')
                for file_n in list(files_to_process(
                        files_list_lang[base],
                        output + os.sep + base)):
                    file.write('\n' + base + ',' + file_n)
