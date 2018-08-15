"""
This script generates spectrogram of wav audio files using parallel processing.
"""

from preprocessing.spectrogram import *
from tqdm import tqdm
from util.timing import Timer
import concurrent.futures
import argparse
import warnings
import time
import sys
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def files_to_process(files_list, output_dir):
    this_files = files_list.copy()
    print('Checking files in ', output_dir)
    counter = 0
    # List the output files in the directory
    output_files = os.listdir(output_dir)
    for file_name in files_list:
        if str(file_name[:-3]) + 'png' in output_files:
            this_files.remove(file_name)
            specs_imgs += 1
    print('There are {} files remaining to process, '
          'and {} spectrogram images in {}.'.format(len(files_list)
                                                    - counter,
                                                    counter,
                                                    output_dir))
    return this_files


def process_files_in_parallel(data_path: str, plot_path: str, files_list: list,
                              fun: callable, workers=None, **kwargs):
    """
    Process files in parallel, calling the provided function.

    Expected function signature: audiopath: str, plotpath: str, name: str,
        **kwargs

    ..see: preprocessing.spectrogram for more details.

    :param data_path: str
        Source of audio files.
    :param plot_path: str
        Output directory for plot images.
    :param files_list: list
        List of files to process.
    :param fun:
        A function to call. Default to preprocessing.spectrogram function.
    :param workers: int
        The maximum number of processes that can be used to execute the given
        calls
    :param kwargs:
        Additional kwargs are passed on to the callable object.
    """
    if len(files_list) == 0:
        return
    else:
        print('Processing files through function', fun.__name__, kwargs)
    with concurrent.futures.ProcessPoolExecutor(workers) \
            as executor:
        futures = [executor.submit(fun, audiopath=data_path + os.sep + file_n,
                                   plotpath=plot_path,
                                   name=file_n[:-4],
                                   **kwargs)
                   for file_n in files_list]

        kwargs = {
            'total': len(futures),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
            pass
        with open('logs/scripts/script_specs-exceptions.txt', 'a') as log:
            log.write('Exceptions for {} call at {}'.format(fun.__name__,
                                                            time.time()))
            for f in futures:
                if f.exception() is not None:
                    log.write('\n' + str(f.exception()) +
                              ' \n\twhen processing ' + f)


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generate spectrogram images '
                                                 'of wav audio files.')
    parser.add_argument('source', help='Source directory.')
    parser.add_argument('output', help='Output directory.')
    parser.add_argument('--workers', help='Define how many process to run in '
                                          'parallel.', default=4, type=int)
    parser.add_argument('--check', help='Check output directories ignoring '
                                        'files already processed.',
                        action='store_true')
    parser.add_argument('--check_after', help='Check output directories '
                                              'after processing files and save '
                                              'a list of failed files.',
                        action='store_true')
    os.makedirs('logs/scripts/', exist_ok=True)

    # Set source and output directories:
    arguments = parser.parse_args()
    data_dir = arguments.source
    output = arguments.output

    # Make directories
    os.makedirs(output + '/dft', exist_ok=True)
    os.makedirs(output + '/sox', exist_ok=True)
    os.makedirs(output + '/librosa/default', exist_ok=True)
    os.makedirs(output + '/librosa/log', exist_ok=True)
    os.makedirs(output + '/librosa/mel', exist_ok=True)
    os.makedirs(output + '/librosa/default/log', exist_ok=True)
    os.makedirs(output + '/librosa/default/mel', exist_ok=True)
    os.makedirs(output + '/matplotlib', exist_ok=True)

    files = os.listdir(data_dir)

    # Check processed files to ignore them
    if arguments.check:
        print("Checking, this might take a while... ")
        files_dft = files_to_process(files, output + '/dft')
        files_sox = files_to_process(files, output + '/sox')
        files_lbr_dft = files_to_process(files, output + '/librosa/default')
        files_lbr_log = files_to_process(files, output + '/librosa/log')
        files_lbr_mel = files_to_process(files, output + '/librosa/mel')
        files_lbr_dm = files_to_process(files, output + '/librosa/default/log')
        files_lbr_dl = files_to_process(files, output + '/librosa/default/mel')
        files_plt = files_to_process(files, output + '/matplotlib')
    else:
        files_dft = files
        files_sox = files
        files_lbr_dft = files
        files_lbr_log = files
        files_lbr_mel = files
        files_lbr_dm = files
        files_lbr_dl = files
        files_plt = files

    with Timer() as timer:
        process_files_in_parallel(fun=specgram_dft, files_list=files_dft,
                                  data_path=data_dir, plot_path=output + '/dft',
                                  workers=arguments.workers)
        process_files_in_parallel(fun=specgram_sox, files_list=files_sox,
                                  data_path=data_dir, plot_path=output + '/sox',
                                  workers=arguments.workers)
        process_files_in_parallel(fun=specgram_lbrs, files_list=files_lbr_dft,
                                  data_path=data_dir,
                                  plot_path=output + '/librosa/default',
                                  workers=arguments.workers)
        process_files_in_parallel(fun=specgram_lbrs, files_list=files_lbr_log,
                                  data_path=data_dir, plot_path=output +
                                  '/librosa/log', algorithm='log',
                                  workers=arguments.workers)
        process_files_in_parallel(fun=specgram_lbrs, files_list=files_lbr_mel,
                                  data_path=data_dir, plot_path=output +
                                  '/librosa/mel', algorithm='mel',
                                  workers=arguments.workers)
        process_files_in_parallel(fun=specgram_lbrs, files_list=files_lbr_dl,
                                  data_path=data_dir, plot_path=output +
                                  '/librosa/default/log', y_axis='log',
                                  workers=arguments.workers)
        process_files_in_parallel(fun=specgram_lbrs, files_list=files_lbr_dm,
                                  data_path=data_dir, plot_path=output +
                                  '/librosa/default/mel', y_axis='mel',
                                  workers=arguments.workers)
        process_files_in_parallel(fun=specgram_mtplt, files_list=files_plt,
                                  data_path=data_dir,
                                  plot_path=output + '/matplotlib',
                                  workers=arguments.workers)

    print('WORK DONE, total taken time: %.03f sec.' % timer.interval)
