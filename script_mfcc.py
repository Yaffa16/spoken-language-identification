"""
This script generates mfcc data of wav audio files using parallel processing.
"""
# todo: ignore large files
from tqdm import tqdm
from util.timing import Timer
import numpy as np
import concurrent.futures
import argparse
import librosa
import warnings
import time
import sys
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def files_to_process(files_list: list, output_dir: str,
                     contains_extension=True, is_base_name=True) -> list:
    """
    Build a list of remaining files to process.

    This function checks the output directories searching for generated
    spectrogram images.

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
    print('Checking files in', output_dir)
    counter = 0
    # List the output files in the directory
    output_files = os.listdir(output_dir)

    # Iterate over file names and eliminate from this_files if the respective
    # spectrogram already exists.
    for el in files_list:
        if not is_base_name:
            file_name = os.path.basename(el)
        else:
            file_name = el
        if contains_extension:
            ext = file_name.split('.')[-1]
            file_name = str(file_name[:-len(ext)-1])
        if file_name + '.npy' in output_files:
            remaining_files.remove(el)
            counter += 1
    print('There are {} files remaining to process, '
          'and {} mfcc data in {}.'.format(len(files_list) - counter, counter,
                                           output_dir))
    return remaining_files


def process_files_in_parallel(data_path: str, output_path: str,
                              files_list: list, fun: callable, workers=None,
                              **kwargs):
    """
    Process files in parallel, calling the provided function.

    :param data_path: str
        Source of audio files.
    :param output_path: str
        Output directory for plot images.
    :param files_list: list
        List of files to process.
    :param fun:
        A function to call.
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
        futures = [executor.submit(fn=fun,
                                   audio_path=data_path + os.sep + file_n,
                                   output_path=output_path,
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
        with open('logs/scripts/script_mfcc-exceptions.txt', 'a') as log:
            log.write('\nExceptions for {} call at {}'.format(fun.__name__,
                                                              time.asctime()))
            for f in futures:
                if f.exception() is not None:
                    log.write('\n' + str(f.exception()))


# Helper functions (necessary to use concurrent.futures)

def extract_and_save_mfcc(audio_path: str, output_path: str, name: str,
                          duration: int, **kwargs):
    """
    Extract and save mfcc features of an audio file in a binary file in NumPy
    .npy format.

    :param audio_path: str
        Path to the audio file.
    :param output_path: str
        Output path to save the file.
     :param duration: int
        Duration to load up each audio.
    :param name: str
        Output file name.
    :param kwargs: Additional kwargs are passed on to the librosa.load function.
    """
    y, sr = librosa.load(audio_path, duration=duration, **kwargs)
    features = librosa.feature.mfcc(y, sr)
    np.save(file=output_path + os.sep + name, arr=features)


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generate binary array files'
                                                 'of the extracted mfcc '
                                                 'features of wav audio files.')
    parser.add_argument('source', help='Source directory.')
    parser.add_argument('output', help='Output directory.')
    parser.add_argument('--duration', help='Duration to load up each audio.',
                        type=int, default=4)
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
    dur = arguments.duration

    # Make directories
    os.makedirs(output, exist_ok=True)

    files = os.listdir(data_dir)

    # Check processed files to ignore them
    if arguments.check:
        print("\n> CHECKING")
        files = files_to_process(files, output)
    else:
        files = files

    print('\n> PROCESSING')
    with Timer() as timer:
        process_files_in_parallel(fun=extract_and_save_mfcc, duration=dur,
                                  files_list=files, data_path=data_dir,
                                  output_path=output, workers=arguments.workers)

    print('WORK DONE, total taken time: %.03f sec.' % timer.interval)

    if arguments.check_after:
        print("\n> CHECKING")
        files = files_to_process(files, output)

        with open('logs/scripts/mfcc_remaining_files.csv', 'w') as file:
            for rem_file in files:
                file.write('\n' + rem_file)
