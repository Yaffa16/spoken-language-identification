"""
This script creates usable dataloader sets for training. All dataloader of the original
bases will be concatenated before processing.

Dependencies:
    - sox: this module uses the sox software to convert audio files to wav
    format. Please be sure you have this software installed in your PATH system
    variable. More details in http://sox.sourceforge.net
"""

import concurrent.futures
import os
import numpy as np
import time
from tqdm import tqdm
from pydub import AudioSegment
from shutil import copyfile
from shutil import rmtree
from util import syscommand


def update_list_fixing(file_list: list, num_workers: int,
                       output_path_fixed_files: str='temp',
                       target_rate: int=16000, channels=1, output_path=None,
                       min_duration: int=None, verbose_level=0):
    """
    Generates a new list of files to process containing audio files of same
    rate and channels.
    This function will iterate over the list of audio files and the files with
    the incorrect rate or number of channels will be converted.

    The new files will be available in the output path.

    :param file_list: list
        Original list of files.
    :param output_path_fixed_files: str
        Output path of new files created. Default to temp.
    :param target_rate: int
        The target rate of audio files.
    :param channels: int.
        Number of target channels. Default to 1.
    :param num_workers: int
        The maximum number of processes that can be used to execute the tasks.
    :param output_path: str
        Output path of all files. Default to None. If not none, all the files
        will be available in the provided directory, including the files with
        the correct rate and number of channels.
        If an output_path is provided, the output_path_fixed_files will be
        ignored.
    :param min_duration: int
        Set a minimum length to update the list of files. The audio files with
        less than the minimum provided will be ignored.
    :param verbose_level: int
        Verbosity level. See sox for more information.

    :return: list
        An updated list of paths.
    """
    os.makedirs(output_path_fixed_files, exist_ok=True)
    print('[INFO] generating a new list of files (target rate={}, channels={})'
          .format(target_rate, channels))
    new_file_list = file_list.copy()
    if output_path is not None:
        output_path_fixed_files = output_path

    if str(verbose_level) == '2' and num_workers == 1:
        # This code is duplicated for debugging purposes
        for file_path in file_list:
            try:
                nfp, fp = fix_file(file_path, output_path_fixed_files,
                                   target_rate, channels, output_path,
                                   min_duration, verbose_level)
                if nfp is not None:
                    new_file_list.append(nfp)
                    new_file_list.remove(fp)
            except ValueError as error:
                print(error, file_path)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
                as executor:
            futures = [
                executor.submit(fn=fix_file,
                                file_path=file_path,
                                output_path_fixed=output_path_fixed_files,
                                target_rate=target_rate,
                                channels=channels,
                                output_path=output_path,
                                min_duration=min_duration,
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
            for f in futures:
                if f.result() is not None:
                    nfp, fp = f.result()
                    if nfp is not None:
                        new_file_list.append(nfp)
                        new_file_list.remove(fp)
    return new_file_list


def concat(file_list: list, output_dir: str, chunk_size: int=40,
           num_workers: int=None, name: str=None, rate=None,
           trim_silence_threshold: float=0, min_duration: int=None,
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
    :param name: str
        Output name (concatenated audio).
    :param rate: int
        Target rate of audio file.
    :param min_duration: int
        Set a minimum length to concat the list of files. The audio files with
        less than the minimum provided will be ignored. Default to None, all
        files will be processed.
    :param num_workers: int
        The maximum number of processes that can be used to execute the tasks.
    :param exist_ok: bool
        If the target name already exists, raise an FileExistsError if exist_ok
        is False. Otherwise the file will be replaced.
    :param trim_silence_threshold: float.
        Threshold value to trim silence from audio. Default to None, no trim
        will be performed.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    # Todo: remove trim silence feature from this function to a new function
    if len(file_list) == 0:
        raise ValueError('Not possible to process an empty list of files.')

    os.makedirs(output_dir, exist_ok=True)
    file_list = update_list_fixing(file_list, target_rate=rate, channels=1,
                                   min_duration=min_duration,
                                   verbose_level=verbose_level,
                                   num_workers=num_workers)
    print('[INFO] creating dataloader set [%s]' % output_dir)

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

        if str(verbose_level) == '2' and workers == 1:
            # This code is duplicated for debugging purposes
            for chunk in chunks:
                temp_file = concat_chunks(file_list=chunk,
                                          output_path=output_dir + os.sep,
                                          verbose_level=verbose_level)
                if os.path.isfile(output_dir + os.sep + temp_file):
                    # Add file to temp_files:
                    temp_files.add(output_dir + os.sep + temp_file)
                    # Add file to file_list to process again
                    file_list.append(output_dir + os.sep + temp_file)
        else:
            # Make parallel calls to concat chunks
            with concurrent.futures.ProcessPoolExecutor(num_workers) \
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

                for f in tqdm(concurrent.futures.as_completed(futures),
                              **kwargs):
                    pass

                for f in futures:
                    if os.path.isfile(output_dir + os.sep + f.result()):
                        # Add file to temp_files:
                        temp_files.add(output_dir + os.sep + f.result())
                        # Add file to file_list to process again
                        file_list.append(output_dir + os.sep + f.result())
    # Remove temporary files:
    if len(temp_files) == 0:
        print("[FATAL ERROR]: the concatenated file is missing. You might want "
              "to run again with the chunk_size=2, workers=1, and the "
              "verbosity_level=2 parameters for debugging purposes.")
        exit(-1)
    final_file = file_list[0]
    temp_files.remove(final_file)
    for file in temp_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            print('[WARN] File not found:', file)

    if trim_silence_threshold is not None and trim_silence_threshold > 0:
        temp_file = final_file + '_temp_trs.wav'
        cmd = 'sox -V' + str(verbose_level) + ' ' + final_file + ' ' + \
              temp_file + ' silence 1 0.1 {}% -1 0.1 {}%'.\
                  format(trim_silence_threshold, trim_silence_threshold)
        os.system(cmd)
        os.remove(final_file)
        os.rename(temp_file, final_file)

    if name is not None:
        if os.path.isfile(output_dir + os.sep + name + '.wav'):
            if exist_ok:
                os.remove(output_dir + os.sep + name + '.wav')
            else:
                raise FileExistsError
        os.rename(file_list[0], output_dir + os.sep + name + '.wav')

    rmtree(temp_folder)


def trim(input_file: str, output_path: str, trim_interval: int=5,
         num_workers: int=None, verbose_level=0):
    """
    Trims an audio file.

    This function will generate audio files with the original name of the file
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
    duration = float(syscommand.system('soxi -D ' + input_file))
    if duration == 0:
        # For some reason, the soxi command failed with some large files
        # tested. This is an attempt to get the duration in that case.
        import wave
        import contextlib
        with contextlib.closing(wave.open(input_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
    trims = list(np.arange(0, duration, trim_interval))[:-1]
    if str(verbose_level) == '2' and workers == 1:
        # This code is duplicated for debugging purposes
        for t in trims:
            trim_audio(audio_path=input_file, output_path=output_path,
                       name=prefix + '_' + str(t) + '_' + str(duration),
                       position=t, duration=trim_interval,
                       verbose_level=verbose_level)
    else:
        # Make parallel calls to trim the audio file
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
                as executor:
            futures = [
                executor.submit(fn=trim_audio,
                                audio_path=input_file,
                                output_path=output_path,
                                name=prefix + '_' + str(t) + '_' +
                                     str(duration),
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


def normalize(file_list: list, output_path: str, num_workers: int = None,
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

    if str(verbose_level) == '2' and workers == 1:
        # This code is duplicated for debugging purposes
        for file_path in file_list:
            normalize_audio_sox(audio_path=file_path,
                                output_path=output_path,
                                name='n_' + file_path.split(os.sep)[-1],
                                verbose_level=verbose_level)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) \
                as executor:
            futures = [
                executor.submit(fn=normalize_audio_sox,
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
    if str(verbose_level) == '2':
        print('sox -V%s %s %s' % (verbose_level, files_str, output_path +
                                  os.sep + temp_file_name))
    os.system('sox -V%s %s %s' % (verbose_level, files_str, output_path +
                                  os.sep + temp_file_name))
    return temp_file_name


def normalize_audio_sox(audio_path: str, output_path: str, name: str,
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
    os.system('sox -V%s -v 0.8 %s --norm %s' % (verbose_level, audio_path,
                                                output_path + os.sep + name))


def normalize_audio(audio_path: str, output_path: str, name: str):
    """
    Normalizes an audio file using PyDub.

    :param audio_path: str
        Path to the audio file.
    :param output_path: str
        Output path to the normalized audio.
    :param name: str
        Output name.
    """
    sound = AudioSegment.from_file(audio_path + os.sep + name + '.wav',
                                   "wav")
    change_in_d_bfs = (-20.0) - sound.dBFS
    sound = sound.apply_gain(change_in_d_bfs)
    sound.export(output_path + os.sep + name + '.wav', format="wav")


def trim_audio(audio_path: str, output_path: str, name: str, position, duration,
               verbose_level=0):
    """
    Trims an audio file using sox software.

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


def fix_file(file_path: str, output_path_fixed: str, target_rate: int=16000,
             channels: int=1, output_path: str=None, min_duration: int=None,
             verbose_level: int=0):
    """
    Perform fixes in the provided file if the rate or number of channels
    expected differs from the target rate and number of channels provided.

    :param file_path: str
        The path to the audio file.
    :param output_path_fixed:
        The output path to save the file.
    :param target_rate: int
        The expected rate. Default to 16000.
    :param channels: int
        The expected number of channels. Default to 1.
    :param output_path:
        The output path for all files. Default to None. If not None and the file
        is in the expected format, the file will be copied to the provided path.
        Useful to put all the audio files in the same directory.
    :param min_duration: int
        Set a minimum length to consider. The audio file with less than the
        minimum provided will be ignored. Default to None.
    :param verbose_level: int.
        Verbosity level. Default to 0.

    :return: tuple
        Return a tuple with the path to the audio file in the expected format
        or None if is not possible to fix the file, and the path of the original
        file.
    """
    file_name = file_path.split(os.sep)[-1]
    new_file_path = output_path_fixed + os.sep + file_name

    try:
        # Get the audio length
        out = syscommand.system('soxi -D ' + file_path)
        duration = float(out)
        # Get the number of channels
        out = syscommand.system('soxi -c ' + file_path)
        current_n_channels = int(out)
        # Get the current rate
        out = syscommand.system('soxi -r ' + file_path)
        current_rate = int(out)
        if min_duration is not None and duration < min_duration:
            raise Exception("Minimum length not satisfied")
    except Exception as err:
        if str(verbose_level) != '0':
            print(err)
        return None, file_path

    if current_rate != target_rate and current_n_channels != channels:
        speed = float(current_rate) / float(target_rate)
        cmd = 'sox -V{} -r 16k {} {} channels 1 ' \
              'speed {}'.format(verbose_level, file_path, new_file_path,
                                speed)
        if str(verbose_level) == '2':
            print(cmd)
        os.system(cmd)
        return new_file_path, file_path
    elif current_rate != target_rate:
        speed = float(current_rate) / float(target_rate)
        cmd = 'sox -V{} -r 16k {} {} ' \
              'speed {}'.format(verbose_level, file_path, new_file_path,
                                speed)
        if str(verbose_level) == '2':
            print(cmd)
        os.system(cmd)
        return new_file_path, file_path
    elif current_n_channels != channels:
        cmd = 'sox -V{} {} {} channels 1'.format(verbose_level, file_path,
                                                 new_file_path)
        if str(verbose_level) == '2':
            print(cmd)
        os.system(cmd)
        return new_file_path, file_path

    # Copy file if output path were provided and the file is not there
    if output_path is not None:
        copyfile(file_path, new_file_path)
    return file_path, file_path


if __name__ == '__main__':
    import glob
    import json
    import argparse
    import importlib
    from collections import defaultdict
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generates a dataset of audio '
                                                 'files in a proper format. '
                                                 'This script will concatenate '
                                                 'audio files, normalize the '
                                                 'generated dataset, and trim '
                                                 'the files to make a new '
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
                        default=5, type=int)
    parser.add_argument('--normalize', help='Normalizes audio files (output).',
                        action='store_true')
    parser.add_argument('--rate', help='Set the target rate. If a rate is '
                                       'provided, the files will be scanned '
                                       'before and converted if necessary.',
                        type=int, default=16000)
    parser.add_argument('--min_length', help='Configure a minimum length to '
                                             'concat audio files '
                                             '(ignore otherwise)',
                        type=int, default=4)
    parser.add_argument('--workers', help='Define how many process to run in '
                                          'parallel.',
                        default=4, type=int)
    parser.add_argument('--chunk_size', help='Chunk size to concat audio '
                                             'files. The final result is the '
                                             'same. This might change the '
                                             'general performance of the '
                                             'concatenation process (only for '
                                             'concatenating task).',
                        default=40, type=int)
    parser.add_argument('--limit', help='Set a limit of files to process. '
                                        'Useful when the data is unbalanced '
                                        'and the process is slow.',
                        type=int)
    parser.add_argument('--trim_silence', help='Volume threshold to trim '
                                               'silence from audio files. '
                                               'Default to 0, no trim will be '
                                               'performed. '
                                               'Recommended value: 1',
                        type=float, default=0)
    parser.add_argument('--del_temp', help='Remove temporary files before '
                                           'processing.',
                        action='store_true')
    parser.add_argument('--v', help='Change verbosity level (sox output)',
                        default=0)

    os.makedirs('logs/scripts/', exist_ok=True)

    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.corpus
    output = arguments.output
    seconds = arguments.seconds
    workers = arguments.workers
    output_rate = arguments.rate
    verbose = arguments.v
    min_length = arguments.min_length
    limit = arguments.limit
    trim_silence = arguments.trim_silence

    temp_folder = 'temp'
    if arguments.del_temp and os.path.isdir('temp'):
        rmtree(temp_folder)

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
                                   bases_json[base]['format'],
                                   recursive=True)

        # Set base samples amount
        bases_json[base]['samples'] = len(all_files_path)
        print('Total of raw files: %d' % len(all_files_path))

        # Set the output directory of each base
        out_dir = output + os.sep + bases_json[base]['lang'] + os.sep

        # Check processed files if necessary, removing them
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
        try:
            os.makedirs(temp_folder)
        except OSError:
            print('Error trying to make the temporary directory. Check if the '
                  'folder already exists.')
            exit(1)

        if limit is not None:
            print('[INFO] Limiting amount of files to process')
            for files_list in files_list_lang:
                files_list_lang[files_list] = files_list_lang[files_list][
                                              :limit]

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
                   rate=output_rate,
                   trim_silence_threshold=trim_silence,
                   min_duration=min_length,
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
