"""
This script creates usable data sets for training.

Dependencies:
    - sox: this module uses the sox software to convert audio files to wav
    format. Please be sure you have this software installed in your PATH system
    variable. More details in http://sox.sourceforge.net
    - FFmpeg: this module uses FFmpeg for audio normalization. Please be sure
    you have this software installed in your PATH system variable. More details
    in http://www.ffmpeg.org/

Note:
    Duplicate files are being ignored.
"""
import concurrent.futures
import os
import time
import sys
import subprocess
from tqdm import tqdm
from shutil import copyfile
from pydub import AudioSegment


def files_to_process(files_list: list, output_dir: str,
                     contains_extension=True, is_base_name=True) -> list:
    """
    Build a list of remaining files to process.

    This function checks the data sets integrity.

    :param files_list: list
        The original list of file names to process.
        Only names is considered by default (is_base_name=True).
    :param output_dir: str
        The path of the datasets set (output directory).
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
        Output directory (datasets set).
    :param file_list: list
        List of files to process (paths).
    :param num_workers: int
        The maximum number of processes that can be used to execute the given
        calls
    :param pre_processing: callable
        Pre process datasets before saving. Default to None, no pre processing
        will be performed.
    :param kwargs:
        Additional kwargs are passed on to the pre processing function.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    print('[INFO] creating dataloader set [%s]' % dataset_dir)

    # Set handler function to process each file
    handler = pre_processing if pre_processing is not None else copyfile

    if num_workers == 1:
        for file_path in file_list:
            handler(file_path, dataset_dir, **kwargs)
        return

    # Process dataloader in parallel
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
                              ' \n\twhen processing ' + str(f))


def pre_process(file_path: str, output_dir: str, name: str=None,
                min_length: int=0, trim_interval: tuple=None, normalize=True,
                trim_silence_threshold: float=None, rate: int=None,
                extra_sox_args: str=None, verbose_level=0):
    """
    Pre process a file. Use this function to handle raw datasets.

    :param file_path: str
        Path of the file to process.
    :param output_dir: str
        Path to save the processed file.
    :param name:
        Output name. Default to None (preserves the original name).
    :param min_length: int
        Minimum length of audio to consider. Skip otherwise.
        Note: requires soxi.
    :param trim_interval: tuple, shape=(start, end)
        Trim audio. Default to None (no trim will be performed).
    :param normalize: bool
        Normalizes audio.
    :param trim_silence_threshold: float.
        Threshold value to trim silence from audio. Default to None, no trim
        will be performed.
    :param rate: int
        Sets the rate of output file. Default to None (no conversion will be
        performed).
    :param extra_sox_args: str
        Extra arguments are passed on as command line arguments.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    # Todo: modularize features and rewrite this function
    if min_length > 0:
        process = subprocess.Popen('soxi -D ' + file_path,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        try:
            audio_length = float(stdout.decode(sys.stdout.encoding))
        except ValueError as error:
            if str(verbose_level) == '2':
                print(error, file_path)
            return

        expected_length = trim_interval[1]
        if audio_length < min_length or (trim_interval is not None and
                                         trim_interval[0] +
                                         expected_length > audio_length):
            # Return if the audio length is less than the minimum length
            # required or if the interval provided is not in the necessary range
            # to make a trim.
            return

    if trim_silence_threshold is not None:
        temp_file_path = output_dir + os.sep + \
                         file_path.split(os.sep)[-1].split('.')[-2] + '.wav'
        cmd = 'sox -V' + str(verbose_level) + ' ' + file_path + ' ' \
              + temp_file_path + \
              ' silence 1 0.1 {}% -1 0.1 {}%'.format(trim_silence_threshold,
                                                     trim_silence_threshold)
        os.system(cmd)
        pre_process(file_path=temp_file_path,
                    output_dir=output_dir,
                    name=name,
                    min_length=min_length,
                    trim_interval=trim_interval,
                    normalize=normalize,
                    trim_silence_threshold=None,
                    rate=rate,
                    extra_sox_args=extra_sox_args,
                    verbose_level=verbose_level)
        if os.path.isfile(temp_file_path):
            os.remove(temp_file_path)
        return

    cmd = 'sox -V' + str(verbose_level) + ' ' + file_path
    if name is None:
        name = file_path.split(os.sep)[-1].split('.')[-2] + '_' + \
               str(min_length)
        if trim_interval is not None:
            name += '_' + str(trim_interval[0]) + '_' + str(trim_interval[1])
        if normalize:
            name += '_norm'
    cmd += ' ' + output_dir + os.sep + name + '.wav '
    if trim_interval is not None:
        cmd += ' trim ' + str(trim_interval[0]) + ' ' + str(trim_interval[1])
    if extra_sox_args is not None:
        cmd += extra_sox_args
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    if normalize:
        sound = AudioSegment.from_file(output_dir + os.sep + name + '.wav',
                                       "wav")
        change_in_d_bfs = (-20.0) - sound.dBFS
        sound = sound.apply_gain(change_in_d_bfs)
        sound.export(output_dir + os.sep + name + '.wav', format="wav")
    if rate is not None:
        # Get the current rate
        process = subprocess.Popen('soxi -r ' + file_path,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        current_rate = float(stdout.decode(sys.stdout.encoding))
        # Convert the sample rate
        cmd = 'sox -V' + str(verbose_level) + ' −r 16k ' + output_dir + \
              os.sep + name + '.wav ' + output_dir + os.sep + name + '_' + \
              str(rate) + '_temp.wav'
        if str(verbose_level) == '2':
            print(cmd)
        os.system(cmd)
        # Remove old file
        os.remove(output_dir + os.sep + name + '.wav')
        # Speed up to the original time
        input_file = output_dir + os.sep + name + '_' + str(rate) + '_temp.wav'
        cmd = 'sox -V' + str(verbose_level) + ' ' + input_file + ' ' + \
              output_dir + os.sep + name + '_' + str(rate) + \
              '.wav speed ' + str((current_rate/float(rate)))
        if str(verbose_level) == '2':
            print(cmd)
        os.system(cmd)
        # Remove old file
        os.remove(input_file)


if __name__ == '__main__':
    import glob
    import json
    import argparse
    import importlib
    from collections import defaultdict
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a dataset of audio '
                                                 'files in a proper format.')
    parser.add_argument('corpus', help='Corpus information (JSON file)')
    parser.add_argument('output', help='Output directory.')
    parser.add_argument('--seconds', help='Length of audio files in seconds.',
                        type=int)
    parser.add_argument('--trim_silence', help='Volume threshold to trim '
                                               'silence from audio files. '
                                               'Default to 0, no trim will be '
                                               'performed. '
                                               'Recommended values: 1 - 5',
                        type=float, default=0)
    parser.add_argument('--rate', help='Set the output rate of audio files.')
    parser.add_argument('--workers', help='Define how many process to run in '
                                          'parallel.', default=4, type=int)
    parser.add_argument('--check', help='Check output directories ignoring '
                                        'files already processed.',
                        action='store_true')
    parser.add_argument('--just_check', help='Check output directories. A csv '
                                             'log file will be created '
                                             'containing the remaining files to'
                                             ' process. No processing will be '
                                             'performed.',
                        action='store_true')
    parser.add_argument('--augment_data', help='Try to augment data by '
                                               'trimming audio files in '
                                               'different positions.',
                        action='store_true')
    parser.add_argument('--limit', help='Set a limit of files to process. '
                                        'Useful when the data is unbalanced '
                                        'and the process is slow.',
                        type=int)
    parser.add_argument('--v', help='Change verbosity level (sox output)',
                        default=0)

    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.corpus
    output = arguments.output
    seconds = arguments.seconds
    augment_data = arguments.augment_data
    output_rate = arguments.rate
    workers = arguments.workers
    verbose = arguments.v
    limit = arguments.limit
    trim_silence = arguments.trim_silence

    # Enable logging the remaining files with just check option.
    if arguments.just_check:
        os.makedirs('logs/scripts/', exist_ok=True)
        log_remaining = open('logs/scripts/dataset_remaining_files.csv', 'w')
        log_remaining.write('base' + ',' + 'original_base' + ',' + 'path')
    else:
        log_remaining = None

    # Load json info about bases
    with open(data_dir) as base_json:
        bases_json = json.load(base_json)

    # The files will be processed as a new base by their language
    files_list_lang = defaultdict(lambda: [])

    # Get a list of files in each language
    for base in bases_json:
        print('\n[INFO] getting a list of files of base "%s"' % base)
        if bases_json[base]['format'] not in ['wav', 'mp3', 'sph', 'flac',
                                              'aiff', 'ogg', 'aac', 'wma']:
            print('[WARN] unknown format of base')
            bases_json[base]['format'] = '*'

        # Get a list of all files (paths) to process
        all_files_path = glob.glob(bases_json[base]['path'] + '/**/*.' +
                                   bases_json[base]['format'], recursive=True)

        # Set base samples amount
        bases_json[base]['samples'] = len(all_files_path)
        print('Total of raw files: %d' % len(all_files_path))

        # Set the output directory of each base
        out_dir = output + os.sep + bases_json[base]['lang'] + os.sep

        # Check processed files if necessary, removing them
        if arguments.check or arguments.just_check:
            files_paths = files_to_process(all_files_path, out_dir,
                                           contains_extension=True,
                                           is_base_name=False)
        else:
            files_paths = None

        # Set the new base:
        new_base_name = bases_json[base]['lang']

        if log_remaining is not None and files_paths is not None:
            for f in files_paths:
                log_remaining.write('\n' + new_base_name + ',' + base + ',' + f)

        else:
            files_paths = all_files_path

        # Append file paths to the respective language base
        files_list_lang[new_base_name] += files_paths

    if limit is not None:
        print('[INFO] Limiting amount of files to process')
        for files_list in files_list_lang:
            files_list_lang[files_list] = files_list_lang[files_list][:limit]

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
        print(str(pd.DataFrame(d)) + '\n')

    if arguments.just_check:
        exit(0)

    # Process files of each language as a new base:
    for base in files_list_lang:
        if len(files_list_lang[base]) == 0:
            continue
        print('[INFO] processing base "%s"' % base)
        # Call function and make data set
        # Create attempts to trim in different positions. A better approach to
        # this would be making the trims considering the audio length, but this
        # approach is good for now.
        if augment_data and seconds is not None:
            trims = 8
        else:
            trims = 1
        for d in range(0, trims, 2):
            print('[INFO] operation {} of {}'.format(int(d/2),
                                                     int(trims/2 - 1)))
            create_dataset(dataset_dir=output + os.sep + base,
                           file_list=files_list_lang[base],
                           num_workers=workers,
                           pre_processing=pre_process,
                           verbose_level=verbose,
                           min_length=seconds if seconds is not None else 0,
                           rate=output_rate,
                           trim_silence_threshold=trim_silence,
                           trim_interval=(0 + d, seconds)
                           if seconds is not None else None)
