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
import time
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


def concat(file_list: list, output_dir: str, chunk_size: int=40,
           num_workers: int=None, name: str=None,
           exist_ok=False, verbose_level=0):
    """
    Concat a list of audio files in one file.

    :param file_list: list
        List of files to process (paths).
    :param output_dir: str
        Output directory.
    :param chunk_size: int
        Chunk size to concat audio files. The final result is the same. This
        can change the general performance of the concatenation process.
    :param name:
        Output name (concatenated audio).
    :param num_workers: int
        The maximum number of processes that can be used to execute the tasks.
    :param exist_ok: bool
        If the target name already exists, raise an FileExistsError if exist_ok
        is False. Otherwise the file will be replaced.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    if len(file_list) == 0:
        raise ValueError('Not possible to process an empty list of files.')

    os.makedirs(output_dir, exist_ok=True)
    print('[INFO] creating data set [%s]' % output_dir)

    # Temp files (will be removed after concatenation process)
    temp_files = set()

    # Concat all files
    print('[INFO] concatenating chunks of size %d' % chunk_size)
    while len(file_list) > 1:  # Will concat chunk of files until lasts one left
        print('[remaining files to process: %d]' % len(file_list))
        # Create chunks
        chunks = [file_list[i:i + chunk_size]
                  for i in range(0, len(file_list), chunk_size)]

        # Reset file list
        file_list = []

        # Make parallel calls to concat chunks
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
                as executor:
            futures = [
                executor.submit(fn=concat_chunks,
                                file_list=chunk,
                                output_path=output_dir + os.sep,
                                verbose_level=verbose_level)
                for chunk in chunks]

            kwargs = {
                'total': len(futures),
                'unit': 'chunks',
                'unit_scale': True,
                'leave': True
            }

            for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
                pass

            for f in futures:
                # Add file to temp_files:
                temp_files.add(output_dir + os.sep + f.result())
                # Add file to file_list to process again
                file_list.append(output_dir + os.sep + f.result())

    # Remove temporary files:
    temp_files.remove(file_list[0])
    for file in temp_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            print('[WARN] File not found:', file)

    if name is not None:
        if os.path.isfile(output_dir + os.sep + name + '.wav'):
            if exist_ok:
                os.remove(output_dir + os.sep + name + '.wav')
            else:
                raise FileExistsError
        os.rename(file_list[0], output_dir + os.sep + name + '.wav')


def trim(input_file: str, output_path: str, trim_interval: int=5,
         num_workers: int = None, verbose_level=0):
    """
    Trim an audio file.

    This function will generated audio files with the original name of the file
    + position which the audio begins and the duration in seconds of the audio
    (for instance file_5.0_10.0.wav refers to an audio which original name is
    'file', with 10 seconds of duration, and the start position begins at 5
    seconds in the original file).

    :param input_file: str
        Path to the file to process.
    :param output_path: str
        Output path for the generated files.
    :param trim_interval: tuple, shape=(start, end)
        Trim audio. Default to None (no trim will be performed).
    :param num_workers: int
        The maximum number of processes that can be used to execute the tasks.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    prefix = input_file.split(os.sep)[-1][:-4]  # Get name and remove extension
    audio = wave.open(input_file, 'r')
    duration = audio.getnframes() / float(audio.getframerate())
    audio.close()
    trims = list(np.arange(0, duration, trim_interval))
    # trims = list(zip(trims[0:len(trims) - 1], trims[1:len(trims)]))

    # Make parallel calls to trim the audio file
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
            as executor:
        futures = [
            executor.submit(fn=trim_audio,
                            audio_path=input_file,
                            output_path=output_path,
                            name=prefix + '_' + str(t) + '_' + str(duration),
                            position=t,
                            duration=trim_interval,
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


def normalize(file_list: list, output_path :str, num_workers: int = None,
              verbose_level=0):
    """
    Creates a normalized dataset.
    This function will generated audio files with the original name of the file
    + the prefix n_.

    :param file_list: list
        List of files to process (paths).
    :param num_workers: int
        The maximum number of processes that can be used to execute the tasks.
    :param output_path: str
        Output path for the generated files.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    if len(file_list) == 0:
        raise ValueError('Not possible to process an empty list of files.')

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
            as executor:
        futures = [
            executor.submit(fn=normalize_audio,
                            audio_path=file_path,
                            output_path=output_path,
                            name='n_' + file_path.split(os.sep)[-1],
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


# Helper functions (necessary to use concurrent.futures)

def concat_chunks(file_list: list, output_path: str, verbose_level=0) -> str:
    """
    Concat chunks of audio files using sox software.

    For each chunk,

    :param file_list: list
        List of files to process.
    :param output_path: str
        Output path to the concatenated file.
    :param verbose_level: int
        Verbosity level. See sox for more information.

    :return: str
        The name of the concatenated file (useful to keep track of temporary
        files).
    """
    temp_file_name = 'temp_' + str(len(file_list)) + \
                     str(int(round(time.time() * 1000))) + '.wav'
    files_str = ' '.join(file_list)
    os.system('sox -V%s %s %s' % (verbose_level, files_str, output_path +
                                  os.sep + temp_file_name))
    return temp_file_name


def normalize_audio(audio_path: str, output_path: str, name: str,
                    verbose_level=0):
    """
    Normalizes an audio file using sox software.

    :param audio_path: str
        Path to the audio file.
    :param output_path: str
        Output path to the normalized audio.
    :param name: str
        Output name.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    os.system('sox -V%s %s --norm %s' % (verbose_level, audio_path,
                                         output_path + os.sep + name))


def trim_audio(audio_path: str, output_path: str, name: str, position, duration,
               verbose_level=0):
    """
    Trim audio files using sox software.

    :param audio_path: str
        Path to the audio to trim.
    :param output_path: str
        Output path to the trimmed audio.
    :param name: str
        Output name.
    :param position:
        Start position to trim.
    :param duration:
        Duration in seconds
    :param verbose_level:
        Verbosity level. See sox for more information.
    """
    os.system('sox -V%s %s %s.wav trim %s %s' % (verbose_level, audio_path,
                                                 output_path + os.sep + name,
                                                 position, duration))


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a dataset of audio '
                                                 'files in a proper format. '
                                                 'This script will concatenate '
                                                 'audio files, normalize the '
                                                 'generated audio, and trim '
                                                 'the audio to make a new '
                                                 'dataset of audio files. '
                                                 'To change this default '
                                                 'behavior, select the tasks '
                                                 'to execute (useful if the '
                                                 'execution of the program is '
                                                 'interrupted for some '
                                                 'reason).')
    parser.add_argument('corpus', help='Corpus information (JSON file)')
    parser.add_argument('output', help='Output directory.')
    parser.add_argument('--concat', help='Concat audio files (source) '
                                         'in a single file '
                                         '(output/corpus_name).',
                        action='store_true')
    parser.add_argument('--trim', help='Trim audio files (output)',
                        action='store_true')
    parser.add_argument('--seconds', help='Time interval to trim audio files '
                                          '(only for trim task).',
                        default=5)
    parser.add_argument('--normalize', help='Normalizes audio files (output).',
                        action='store_true')
    parser.add_argument('--workers', help='Define how many process to run in '
                                          'parallel.',
                        default=4, type=int)
    parser.add_argument('--check', help='Check output directories ignoring '
                                        'files already processed.',
                        action='store_true')
    parser.add_argument('--chunk_size', help='Chunk size to concat audio '
                                             'files. The final result is the '
                                             'same. This might change the '
                                             'general performance of the '
                                             'concatenation process (only for '
                                             'concatenating task).',
                        default=40, type=int)
    parser.add_argument('--v', help='Change verbosity level (sox output)',
                        default=0)

    os.makedirs('logs/scripts/', exist_ok=True)

    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.corpus
    output = arguments.output
    seconds = arguments.seconds
    workers = arguments.workers
    verbose = arguments.v

    # Creating a dict to store which operations it will be performed.
    perform_task = dict()
    # The operations will be executed in the following order:
    # - concat files from the path provided by the source (json) file
    # - trim files
    # - normalize files
    # Each subsequent task needs to be performed after the previous task
    # (no need to execute in a single program call though).
    perform_task['concat'] = arguments.concat
    perform_task['trim'] = arguments.trim
    perform_task['normalize'] = arguments.normalize
    if not perform_task['concat'] and not perform_task['trim'] and not \
            perform_task['normalize']:
        for task in perform_task:
            perform_task[task] = True
    perform_task['check'] = arguments.check

    # Load json info about bases
    with open(data_dir) as base_json:
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
        if perform_task['check']:
            files_paths = files_to_process(all_files_path, out_dir,
                                           contains_extension=True,
                                           is_base_name=False)
        else:
            files_paths = all_files_path

        # Append file paths to the respective language base
        files_list_lang[bases_json[base]['lang']] += files_paths

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
        print(str(pd.DataFrame(d)))

    if perform_task['concat']:
        print('\n> CONCATENATING FILES')
        print('TOTAL FILES TO BE PROCESSED: %d' %
              (sum(len(b) for b in files_list_lang.values())))
        for base in files_list_lang:
            print('[INFO] processing base "%s"' % base)

            # Call function and concat original corpus in a single file
            concat(output_dir=output + os.sep + base,
                   file_list=files_list_lang[base],
                   num_workers=workers,
                   verbose_level=verbose,
                   name=base,
                   exist_ok=True,
                   chunk_size=arguments.chunk_size)

    if perform_task['trim']:
        # Update list of files to process
        to_process = dict()
        for base in files_list_lang:
            to_process[base] = output + os.sep + base + os.sep + base + '.wav'
        print('\n> TRIMMING AUDIO FILES')
        for base in files_list_lang:
            print('[INFO] processing base "%s"' % base)
            # Call function and trim audio files
            trim(input_file=to_process[base],
                 output_path=output + os.sep + base,
                 num_workers=workers,
                 verbose_level=verbose,
                 trim_interval=seconds)

        # Remove old files:
        for base in to_process:
            os.remove(to_process[base])

    if perform_task['normalize']:
        # Update list of files to process
        to_process = dict()
        for base in files_list_lang:
            os.makedirs(output + os.sep + base + '_normalized', exist_ok=True)
            to_process[base] = glob.glob(output + os.sep + base + os.sep +
                                         '*.wav')
            print('\n> NORMALIZING FILES')
            # Call function and normalize audio files
            normalize(file_list=to_process[base],
                      output_path=output + os.sep + base + '_normalized',
                      num_workers=workers,
                      verbose_level=verbose)
