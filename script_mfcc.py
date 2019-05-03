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
                              n_mfcc: int=20, verbose_level: int=0, **kwargs):
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
        calls.
    :param n_mfcc: int > 0 [scalar]
        number of MFCCs to return.
    :param verbose_level: int
        Verbose level
    :param kwargs:
        Additional kwargs are passed on to the callable object.
    """
    if len(files_list) == 0:
        return
    else:
        print('[INFO] Processing files through function', fun.__name__, kwargs)

    with concurrent.futures.ProcessPoolExecutor(workers) \
            as executor:
        futures = [executor.submit(fn=fun,
                                   audio_path=data_path + os.sep + file,
                                   output_path=output_path,
                                   name=os.path.basename(file)[:-4],
                                   n_mfcc=n_mfcc,
                                   verbose_level=verbose_level,
                                   **kwargs)
                   for file in files_list]

        kw = {
            'total': len(futures),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kw):
            pass
        with open('logs/scripts/script_mfcc-exceptions.txt', 'a') as log:
            log.write('\nExceptions for {} call at {}'.format(fun.__name__,
                                                              time.asctime()))
            for f in futures:
                if f.exception() is not None:
                    log.write('\n' + str(f.exception()))


# Helper functions (necessary to use concurrent.futures)

def extract_and_save_mfcc(audio_path: str, output_path: str, name: str,
                          duration: int, n_mfcc: int=20, verbose_level: int=0,
                          **kwargs):
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
    :param n_mfcc: int > 0 [scalar]
        number of MFCCs to return.
    :param verbose_level: int
        Verbose level
    :param kwargs: Additional kwargs are passed on to the librosa.load function.
    """
    if verbose_level > 1:
        print('[INFO] processing file {}'.format(audio_path))
    y, sr = librosa.load(audio_path, duration=duration, **kwargs)
    features = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc)
    if verbose_level > 1:
        print('[INFO] features extracted')
        print('[INFO] saving file {}.npy'.format(os.path.join(output_path,
                                                              name)))
    if verbose_level > 2:
        print('[FEATURES] > {}'.format(y, sr))
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
    parser.add_argument('--n_mfcc', help='Number of MFCCs to return.',
                        type=int, default=20)
    parser.add_argument('--workers', help='Define how many process to run in '
                                          'parallel.', default=4, type=int)
    parser.add_argument('--check', help='Check output directories ignoring '
                                        'files already processed.',
                        action='store_true')
    parser.add_argument('--check_after', help='Check output directories '
                                              'after processing files and save '
                                              'a list of failed files.',
                        action='store_true')
    parser.add_argument('-R', help='Process files recursively. Note: the '
                                   'processed files will be saved preserving '
                                   'the folder name of the raw files. '
                                   'For example, path/to/the/folder/file.wav '
                                   'will be saved as '
                                   'output_dir/folder/file.npy .',
                        action='store_true', default=False)
    parser.add_argument('--v', help='Verbose level', type=int, default=0)
    parser.add_argument('--only_wav', help='Force to process only files '
                                           'with .wav extension.',
                        action='store_true', default=False)

    os.makedirs('logs/scripts/', exist_ok=True)

    # Set source and output directories:
    arguments = parser.parse_args()
    data_dir = arguments.source
    output = arguments.output
    dur = arguments.duration
    mfcc = arguments.n_mfcc
    rec = arguments.R
    v_level = arguments.v
    only_wav = arguments.only_wav

    # Make directories
    if rec:
        if v_level > 1:
            print('[INFO] recursive mode enabled')
        folders = []
        for r, d, f in os.walk(data_dir):
            for folder in d:
                if v_level > 1:
                    print('[INFO] will process directory '
                          '{}'.format(os.path.join(r, folder)))
                folders.append(os.path.join(r, folder))
                if not os.path.isdir(os.path.join(output, folder)):
                    os.makedirs(os.path.join(output, folder), exist_ok=True)
                    if v_level > 1:
                        print('[INFO] creating directory '
                              '{}'.format(os.path.join(output, folder)))
    else:
        if v_level > 1:
            print('[INFO] will process directory {}'.format(data_dir))
            print('[INFO] creating directory {}'.format(output))
        os.makedirs(output, exist_ok=True)
        folders = [data_dir]

    print('\n> PROCESSING')
    with Timer() as timer:
        for folder in (tqdm(folders, total=len(folders), unit='dirs',
                            unit_scale=True, leave=True)
                       if v_level < 2 else folders):
            if v_level > 1:
                print('[INFO] processing folder {}'.format(folder))
            # Check processed files to ignore them
            if arguments.check:
                print("\n> CHECKING")
                paths = files_to_process(os.listdir(folder),
                                         os.path.join(output,
                                                      os.path.basename(folder)))
            else:
                paths = os.listdir(folder)

            for path in paths:
                if os.path.basename(path)[-3:] != 'wav':
                    if v_level > 1:
                        print('[WARN] non wav file detected: {}'.format(path))
                    if only_wav:
                        print('[INFO] ignoring file {}'.format(path))
                        paths.remove(path)

            if v_level > 2:
                print('[FILES] > {}'.format(paths))
            if arguments.workers == 1:
                for file in (tqdm(paths) if v_level < 2 else paths):
                    extract_and_save_mfcc(audio_path=os.path.join(folder, file),
                                          output_path=os.path.
                                          join(output,
                                               os.path.basename(folder)),
                                          name=os.path.basename(file)[:-4],
                                          duration=dur,
                                          n_mfcc=mfcc,
                                          verbose_level=v_level)
            else:
                process_files_in_parallel(fun=extract_and_save_mfcc,
                                          duration=dur, files_list=paths,
                                          data_path=os.path.
                                          join(data_dir,
                                               os.path.basename(folder)),
                                          n_mfcc=mfcc,
                                          output_path=os.path.
                                          join(output, os.path.
                                               basename(folder)),
                                          workers=arguments.workers,
                                          verbose_level=v_level)

            if arguments.check_after:
                print("\n> CHECKING")
                paths = files_to_process(os.listdir(folder),
                                         os.path.join(output,
                                                      os.path.basename(folder)))

                with open('logs/scripts/mfcc_remaining_files.csv', 'w') as file:
                    for rem_file in paths:
                            file.write('\n' + rem_file)
    print('WORK DONE, total taken time: %.03f sec.' % timer.interval)
