"""
This script creates usable data sets for audio recognition.

Dependencies:
    - sox: this module uses the sox software to convert audio files to wav
    format. Please be sure you have this software installed in your PATH system
    variable. More details in http://sox.sourceforge.net

Note:
    Duplicate files are being ignored.
"""
import concurrent.futures
import os
import time
import glob
import sys
import util.syscommand as syscommand
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment

# 8 speeds between (0.8, 1.2); remove the speed with value 1
SPEEDS = np.delete(np.linspace(0.8, 1.2, 9), 4)

# 8 semitones between (-200, 200); remove the semitone with value 0
SEMITONES = np.delete(np.linspace(-200, 200, 9), 4)

# Changed in 22/03:
# make list of all files in preprocessing/noises
NOISES = list(glob.glob('preprocessing/noises/**/*.wav', recursive=True))

# Changed in 22/03:
# make function get_audio_info


def get_audio_info(file_path, *args, verbose_level=0):
    info = dict()
    value = None
    try:
        # To add new feature:
        # if 'feature' in args:
        #   run the command
        if 'duration' in args:
            value = syscommand.system('soxi -D {file}'.format(file=file_path),
                                      debug=True if int(verbose_level) > 0
                                      else False)
            info['duration'] = float(value)
        if 'rate' in args:
            value = syscommand.system('soxi -r {file}'.format(file=file_path),
                                      debug=True if int(verbose_level) > 0
                                      else False)
            info['rate'] = float(value)
    except ValueError as error:
        if str(verbose_level) == '2':
            print('[ERROR] Trying to get information of file {file}. '
                  '{error}.'.format(file=file_path, error=error))
        raise ValueError('Trying to get information of file {file}'.
                         format(file=file_path))
    # If just one value were requested, return a single value
    return info if len(info.keys()) > 1 else value


def files_to_process(files_list: list, output_dir: str) -> list:
    """
    Build a list of remaining files to process.

    This function checks the data sets integrity.

    :param files_list: list
        The original list of file names to process.
        Only names is considered by default (is_base_name=True).
    :param output_dir: str
        The path of the datasets (output directory).

    :return: list
        A list containing the remaining files to process (in the format of the
        provided list).
    """
    remaining_files = list()
    print('[INFO] checking directories in', output_dir)

    for file in files_list:
        if not os.path.isdir(output_dir + os.sep +
                             os.path.basename(file)[:-4]):
            remaining_files.append(file)

    print('There are {} files remaining to process'.
          format(len(remaining_files)))
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
        Pre process datasets before saving. Default to None, no pre processing
        will be performed.
    :param kwargs:
        Additional kwargs are passed on to the pre processing function.
    """
    if len(file_list) == 0:
        print('[WARN] no files to process the dataset {dataset}!'.
              format(dataset=dataset_dir))
    print('[INFO] creating data set {dataset}'.format(dataset=dataset_dir))
    os.makedirs(dataset_dir, exist_ok=True)

    if num_workers == 1:
        for file_path in file_list:
            # New feature (changed at 25/03) -> separate files by directory
            # pre_processing(file_path, dataset_dir, **kwargs)
            pre_processing(file_path, dataset_dir + os.sep + '_' +
                           os.path.basename(file_path)[:-4], **kwargs)
        return

    # Process data in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as \
            executor:
        # New feature (changed at 25/03) -> separate files by directory
        # futures = [executor.submit(pre_processing, file_path,
        #                            dataset_dir, **kwargs)
        #            for file_path in file_list]
        futures = [executor.submit(pre_processing, file_path,
                                   dataset_dir + os.sep + '_' +
                                   os.path.basename(file_path)[:-4],
                                   **kwargs)
                   for file_path in file_list]

        kw = {
            'total': len(futures),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kw):
            pass
        with open('logs/scripts/script_create_dataset.txt', 'a') as log:
            log.write('\nExceptions for {function} call at '
                      '{time}'.format(function=pre_processing.__name__,
                                      time=time.time()))
            for f in futures:
                if f.exception() is not None:
                    log.write('\n{exception}\n\twhen processing {file}'
                              .format(exception=str(f.exception()),
                                      file=str(f)))


def pre_process(file_path: str, output_dir: str, name: str=None,
                min_length: int=0, trim_interval: tuple=None, normalize=False,
                trim_silence_threshold: float=None, rate: int=None,
                remix_channels: bool=False, speed_changing: float=None,
                pitch_changing: float=None, noise_path: str=None,
                ignore_length: bool=False, verbose_level=0):
    """
    Pre process a file. Use this function to handle raw datasets.

    This function generates temporary audio files. All files will be removed in
    the end of the process, except the original file provided and the final
    output file.

    :param file_path: str
        Path of the file to process.
    :param output_dir: str
        Path to save the processed file.
    :param name:
        Output name. Default to None (preserves the original name).
    :param min_length: int
        Minimum length of audio to consider. Skip otherwise.
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
    :param remix_channels: bool
        Mix channels generating a mono audio. Default to True. It is1
        recommended to set this argument as True when the rate is converted.
    :param pitch_changing: int
        Integer representing the pitch changing effect applied to the audio.
        This changes the pitch of the audio, but not its speed.
    :param noise_path: str
        Path to an audio representing a noise for mixing with the original file.
    :param speed_changing: float
        Percentage representing the changing of the speed of the audio.
    :param ignore_length: bool
        If true, will pass --ignore_length to each audio, forcing the length
        checking. Can slow down the process.
    :param verbose_level: int
        Verbosity level. See sox for more information.
    """
    # New feature (changed at 25/03) -> separate files by directory
    os.makedirs(output_dir, exist_ok=True)
    if int(verbose_level) > 1:
        print('[INFO] processing {file}'.format(file=file_path))

    # Create a set of temporary files
    temp_files = set()

    if min_length > 0:
        try:
            audio_length = float(get_audio_info(file_path, 'duration',
                                 verbose_level))
        except ValueError:
            return
        if audio_length is None:
            if str(verbose_level) == '2':
                print('[WARN] audio length of file {} is zero?'.
                      format(file_path))
            return

        expected_length = trim_interval[1] - trim_interval[0] if trim_interval \
                          is not None else min_length
        if audio_length < expected_length and ignore_length:
            # Force the audio length checking (can be slow)
            if str(verbose_level) == '2':
                print('[WARN] forcing audio length checking of audio {file}'
                      ': Got {length} seconds'.format(file=file_path,
                                                      length=audio_length))
            tmp = output_dir + os.sep + 'tmp.wav'
            if os.path.isfile(tmp):
                if str(verbose_level) == '2':
                    print('[WARN] {tmp} already exists: {file} will be skipped '
                          'from processing'.format(tmp=tmp, file=file_path))
                return

            # This command call sox to rewrite the header
            cmd = 'sox -V{vlevel} --ignore-length {file} {tmp}'.\
                format(vlevel=verbose_level,
                       file=file_path,
                       tmp=tmp)
            if str(verbose_level) == '2':
                print(cmd)
            os.system(cmd)
            file_path = output_dir + os.sep + os.path.basename(file_path)
            if os.path.isfile(file_path):
                os.remove(file_path)
            os.rename(tmp, file_path)
            temp_files.add(file_path)

            audio_length = float(get_audio_info(file_path, 'duration',
                                                verbose_level))
            if str(verbose_level) == '2':
                print('[INFO] fixed audio length of audio {file}'
                      ': Got {length} seconds'.format(file=file_path,
                                                      length=audio_length))
            if audio_length < expected_length and os.path.isfile(file_path):
                os.remove(file_path)
                if str(verbose_level) == '2':
                    print('[WARN] Length {length} of audio {file} is less than '
                          '{min} (ignoring)'.format(length=audio_length,
                                                    file=file_path,
                                                    min=min_length))
                return

        elif audio_length < expected_length:
            if str(verbose_level) == '2':
                print('[WARN] Length {length} of audio {file} is less than '
                      '{min} (ignoring)'.format(length=audio_length,
                                                file=file_path,
                                                min=min_length))
            return
    else:
        expected_length = None  # Variable not being used. Rare case.
    if name is None:
        # Todo: change file_path.split to os.path.basename(file_path)[:-4]...
        # New feature (changed at 25/03)
        name = file_path.split(os.sep)[-1][:-4] + '_'
    # Process each transformation
    if remix_channels:
        name += '_remix_'
        file_path = remix(file_path, output_dir, name, verbose_level)
        temp_files.add(file_path)
    if rate is not None:
        name += '_rate' + str(rate) + '_'
        file_path = convert_rate(rate, file_path, output_dir, name,
                                 verbose_level)
        temp_files.add(file_path)
    if speed_changing is not None:
        name += '_speed_' + str("%.2f" % speed_changing) + '_'
        file_path = speed(speed_changing, file_path, output_dir, name,
                          verbose_level)
        temp_files.add(file_path)
    if pitch_changing is not None:
        name += '_pitch_' + str(pitch_changing) + '_'
        file_path = pitch(pitch_changing, file_path, output_dir, name,
                          verbose_level)
        temp_files.add(file_path)
    if normalize:
        name += '_norm_'
        file_path = norm(file_path, output_dir, name, verbose_level)
        temp_files.add(file_path)
    if trim_silence_threshold is not None:
        name += '_trs_' + str(trim_silence_threshold) + '_'
        file_path = trim_silence_audio(trim_silence_threshold, file_path,
                                       output_dir, name, verbose_level)
        temp_files.add(file_path)
    if noise_path is not None:
        name += '_noise_' + \
                noise_path.replace('/', os.sep).split(os.sep)[-1]. \
                    split('.')[-2] + '_'
        file_path = add_noise(noise_path, file_path, output_dir, name,
                              verbose_level)
        temp_files.add(file_path)
    if trim_interval is not None:
        name += '_trim_' + str(trim_interval[0]) + '_' + str(
            trim_interval[1]) \
                + '_'
        file_path = trim(file_path, output_dir, name, trim_interval[0],
                         trim_interval[1] - trim_interval[0],
                         verbose_level)
        temp_files.add(file_path)

    # TODO: remove dead code
    # audio_length = float(syscommand.system('soxi -D ' + file_path,
    #                                        debug=True if str(verbose_level)
    #                                                      == '2' else False))
    # if expected_length is None or audio_length >= expected_length:
    #     temp_files.remove(file_path)
    # else:
    #     pass  # The file will be deleted
    #     if int(verbose_level) >= 1:
    #         print('Length of audio {} is less than {} (will be deleted)'.format(
    #             file_path, min_length))
    temp_files.remove(file_path)

    # Remove the temporary files
    for fp in temp_files:
        if os.path.isfile(fp):
            if int(verbose_level) == 2:
                print('[INFO] removing temporary file {}'.format(fp))
            os.remove(fp)


# Helper functions

def norm(file_path: str, output_dir: str, file_name: str, verbose_level: int):
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    sound = AudioSegment.from_file(file_path, "wav")
    change_in_d_bfs = (-20.0) - sound.dBFS
    sound = sound.apply_gain(change_in_d_bfs)
    sound.export(temp_file_path, format="wav")
    return temp_file_path


def trim_silence_audio(trim_threshold: float, file_path, output_dir, file_name,
                       verbose_level: int=0) -> str:
    """
    Removes silence.

    :param trim_threshold: float
        Below-periods duration threshold.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    # todo: add silence threshold argument, in this case, it is set as 1.0
    cmd = 'sox -V{vlevel} {input} {output} '\
          'silence {below_periods_duration} {above_periods} {duration} '\
          '{duration_threshold}% {below_period} {ignore_period} '\
          '{below_period_threshold}%'\
          .format(vlevel=str(verbose_level),
                  input=file_path,
                  output=temp_file_path,
                  below_periods_duration='l',  # Remove silence or short silence
                  # parameter. l means for not remove long periods of silence
                  above_periods='1',  # Start removing silence from the
                  # beginning
                  duration='0.1',  # Minimum duration of silence to remove
                  duration_threshold=trim_threshold,  # Trim silence
                  # (anything less than 1% volume)
                  below_period='-1',  # Remove silence until the end of the file
                  ignore_period='1.0',  # Ignore small moments of silence
                  below_period_threshold=trim_threshold
                  )
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def remix(file_path, output_dir, file_name, verbose_level: int=0) -> str:
    """
    Remix the audio into a single channel audio.

    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} remix 1'.format(vlevel=verbose_level, 
                                                           input=file_path,
                                                           output=temp_file_path
                                                           )
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def convert_rate(rate, file_path, output_dir, file_name,
                 verbose_level: int=0) -> str:
    """
    Remix the audio into a single channel audio.

    :param rate: int
        Integer representing the target rate. Example: 16000.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    # Get the current rate
    # Commented: remove dead code
    # current_rate = float(get_audio_info(file_path, 'rate', verbose_level))

    # Convert the sample rate
    cmd = 'sox -V{vlevel} {input} -r {rate} {output}'.\
          format(vlevel=verbose_level,
                 input=file_path,
                 rate=rate,
                 output=temp_file_path)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)

    # TODO: remove dead code
    # # Speed up to the original time
    # # This is necessary because when the rate is converted in sox the
    # # length of the audio file changes as well.
    # temp_file_speed = speed(current_rate / float(rate), temp_file_path,
    #                         output_dir, temp_file_path.replace('/', os.sep).
    #                         split(os.sep)[-1][:-4] + '_temp',
    #                         verbose_level)
    # os.remove(temp_file_path)
    # os.rename(temp_file_speed, temp_file_path)
    return temp_file_path


def speed(param, file_path, output_dir, file_name, verbose_level: int=0) -> str:
    """
    Speed up the provided audio.

    :param param: float
        Percentage of the speed.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} speed {param}'.\
          format(vlevel=verbose_level,
                 input=file_path,
                 output=temp_file_path,
                 param=param)
    if str(verbose_level) == '2':

        print(cmd)
    os.system(cmd)
    return temp_file_path


def pitch(param, file_path, output_dir, file_name, verbose_level: int=0) -> str:
    """
    Change the audio pitch (but not tempo).

    :param param: int
        Integer representing the semitones to pitch the audio.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} pitch {param}'.\
          format(vlevel=verbose_level,
                 input=file_path,
                 output=temp_file_path,
                 param=param)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def add_noise(noise_path: str, file_path, output_dir, file_name,
              verbose_level: int=0) -> str:
    """
    Adds noise to an audio file.

    :param noise_path: str
        Path to the noise file.
    :param file_path: str
        Path to the file.
    :param output_dir: str
        Output directory.
    :param file_name: str
        Name of the output file.
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument are passed on as
        a command line option in sox arguments.

    :return: str
        Path to the generated file.
    """
    # Check if the duration of the audio is less than the duration of the noise
    info = get_audio_info(file_path, 'duration', 'rate', verbose_level)
    audio_length, rate = float(info['duration']), float(info['rate'])
    noise_length = float(get_audio_info(noise_path, 'duration', verbose_level))

    temp_file_path = output_dir + os.sep + file_name + '.wav'
    if audio_length > noise_length:
        # Process mix with noise repeat1
        cmd = 'sox -V{vlevel} {noise} -r {rate} -p repeat {repeat} |'\
              ' sox -V{vlevel} - -m {input} {output}'.\
              format(vlevel=verbose_level,
                     noise=noise_path,
                     rate=rate,
                     repeat=int(audio_length/noise_length),
                     input=file_path,
                     output=temp_file_path)
    else: 
        # Process mix
        cmd = 'sox -V{vlevel} -m {input} {noise} {output}'.\
              format(vlevel=verbose_level,
                     input=file_path,
                     noise=noise_path,
                     output=temp_file_path)

    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)    

    # Trim to the original time
    temp_file = trim(temp_file_path, output_dir, temp_file_path.
                     replace('/', os.sep).split(os.sep)[-1][:-4]
                     + '_temp', 0, float(get_audio_info(file_path, 'duration',
                                                        verbose_level)))

    if os.path.isfile(temp_file):
        os.remove(temp_file_path)
        os.rename(temp_file, temp_file_path)
    return temp_file_path


def trim(file_path: str, output_dir: str, file_name: str, position, duration,
         verbose_level=0):
    """
    Trims an audio file using sox software.

    :param file_path: str
        Path to the audio to trim.
    :param output_dir: str
        Output path to the trimmed audio.
    :param file_name: str
        Output name.
    :param position:
        Start position to trim.
    :param duration:
        Duration in seconds
    :param verbose_level: int
        Verbosity level. 2 prints the command. This argument as passed on as
        a command line option in sox arguments.
    """
    temp_file_path = output_dir + os.sep + file_name + '.wav'
    cmd = 'sox -V{vlevel} {input} {output} trim {position} {duration}'.\
          format(vlevel=verbose_level,
                 input=file_path,
                 output=temp_file_path,
                 position=position,
                 duration=duration)
    if str(verbose_level) == '2':
        print(cmd)
    os.system(cmd)
    return temp_file_path


def process_augmentation(dataset_dir: str, file_list: list,
                         num_workers: int=None, pre_processing: callable=None,
                         **kwargs):
    print('[INFO] processing augmentation for dataset {dataset}'.
          format(dataset=dataset_dir))
    os.makedirs(dataset_dir, exist_ok=True)

    if num_workers == 1:
        for file_path in file_list:
            pre_processing(file_path,
                           dataset_dir + os.sep +
                           os.path.basename(os.path.dirname(file_path)),
                           **kwargs)
        return

    # Process data in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as \
            executor:
        futures = [executor.submit(pre_processing, file_path,
                                   dataset_dir + os.sep +
                                   os.path.basename(os.path.dirname(file_path)),
                                   **kwargs)
                   for file_path in file_list]

        kw = {
            'total': len(futures),
            'unit': 'files',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kw):
            pass


def augment_data(data_path: str, file_list: list, sliding_window: int=None,
                 trimming_window: int=None, seconds: float=5, noises: list=None,
                 semitones: list=None, speeds: list = None,
                 num_workers: int=None, verbose_level=0, **kwargs):
    """
    Augments data by applying audio transformations.

    See pre_process for more information.

    :param data_path: str
        Path of the dataset. This will be the output path as well.
    :param file_list: list
        List of files to process.
    :param sliding_window: int
        Amount in seconds to slide and trim the audio. Use this for multiple
        small speeches.
    :param trimming_window: int
        Amount in seconds to slide and trim the audio. This process will run
        for each file and will generate fragments of each file depending on
        the length of it. Use this for large audio speeches.
    :param seconds: float
        Length of the audio in seconds to trim. Default to 5.
    :param noises: list
        List of noises paths to apply in each audio. Each noise will generate a
        new audio.
    :param semitones: list
        List of semitones to pitch in each audio. Each semitone will generate
        a new audio.
    :param speeds: list
        List of speeds to apply in each audio. Each speed will generate a new
        audio.
    :param num_workers: int
        Number of workers for multiprocessing .
    :param verbose_level: int
        Verbosity level.
    :param kwargs: dict
        Additional kwargs are passed on to the pre processing function.
    """
    if seconds is None and int(verbose_level) > 0:
        print('[WARN] seconds is not set (length to perform trimming '
              'operations)')
    if semitones is not None:
        print('[INFO] processing pitches')
        for i, p in enumerate(semitones):
            print('[INFO] processing pitches: {} of {}'.
                  format(i + 1, len(semitones)))
            process_augmentation(dataset_dir=data_path,
                                 file_list=file_list,
                                 num_workers=num_workers,
                                 pre_processing=pre_process,
                                 verbose_level=verbose_level,
                                 pitch_changing=p,
                                 **kwargs)
    if speeds is not None:
        print('[INFO] processing speeds')
        for i, s in enumerate(speeds):
            print('[INFO] processing speeds: {} of {}'.format(i + 1,
                                                              len(speeds)))
            process_augmentation(dataset_dir=data_path,
                                 file_list=file_list,
                                 num_workers=num_workers,
                                 pre_processing=pre_process,
                                 verbose_level=verbose_level,
                                 speed_changing=s,
                                 **kwargs)
    if noises is not None:
        print('[INFO] processing noises')
        for i, n in enumerate(noises):
            print('[INFO] processing noises: {} of {}'.format(i + 1,
                                                              len(noises)))
            process_augmentation(dataset_dir=data_path,
                                 file_list=file_list,
                                 num_workers=num_workers,
                                 pre_processing=pre_process,
                                 verbose_level=verbose_level,
                                 noise_path=n,
                                 min_length=seconds,
                                 **kwargs)
    # Get files paths again and process sliding window or trimming to keep
    # a dataset with equal-length audio files
    # file_list = glob.glob(data_path + '/**/*.wav', recursive=True)
    file_list = []
    for dr in os.listdir(data_path):
        if dr[0] == '_':
            file_list += glob.glob(data_path + os.sep + dr + '*/**/*.wav',
                                   recursive=True)
    if sliding_window is not None:
        print('[INFO] processing sliding window')
        for i in range(0, 16, sliding_window):
            print('[INFO] operation {} of {}'.format(int(i / 2) + 1,
                                                     int(16 / 2 - 1) + 1))
            process_augmentation(dataset_dir=data_path,
                                 file_list=file_list,
                                 num_workers=num_workers,
                                 pre_processing=pre_process,
                                 verbose_level=verbose_level,
                                 min_length=seconds,
                                 trim_interval=(0 + i, i + seconds),
                                 **kwargs)
    elif trimming_window is not None: 
        print('[INFO] processing trimming window')
        if workers == 1:
            for audio in (tqdm(file_list) if int(verbose_level) < 2 else
                          file_list):
                if int(verbose_level) > 0:
                    print('[INFO] processing trimming window of audio', audio)

                audio_length = float(get_audio_info(audio, 'duration',
                                     verbose_level))
                for i in range(0, int(audio_length) - trimming_window,
                               trimming_window):
                    if int(verbose_level) > 0:
                        print('[INFO] processing trimming window of {} '
                              '[trimming {} to {}] - [{}/{}]'.
                              format(audio, i, i + seconds, i,
                                     int(audio_length) - trimming_window))

                    # New feature (changed at 25/03) -> separate files by directory
                    # pre_process(output_dir=data_path,
                    #             file_path=audio,
                    #             verbose_level=verbose_level,
                    #             min_length=int(seconds),
                    #             trim_interval=(i, i + seconds),
                    #             **kwargs)
                    pre_process(output_dir=data_path + os.sep +
                                           os.path.basename(os.path.
                                                            dirname(audio)),
                                file_path=audio,
                                verbose_level=verbose_level,
                                min_length=int(seconds),
                                trim_interval=(i, i + seconds),
                                **kwargs)
        else:
            for audio in (tqdm(file_list) if int(verbose_level) < 2 else
            file_list):
                # Process data in parallel
                # New feature (changed at 25/03) -> separate files by directory
                # process_trim_parallel(audio=audio,
                #                       output_dir=data_path,
                #                       seconds=seconds,
                #                       trimming_window=trimming_window,
                #                       num_workers=num_workers,
                #                       verbose_level=verbose_level)
                process_trim_parallel(audio=audio,
                                      output_dir=data_path + os.sep +
                                                 os.path.basename(os.path.
                                                                  dirname(
                                                     audio)),
                                      seconds=seconds,
                                      trimming_window=trimming_window,
                                      num_workers=num_workers,
                                      verbose_level=verbose_level)

    elif seconds is not None:
        print('[INFO] trimming audio files')
        create_dataset(dataset_dir=data_path,
                       file_list=file_list,
                       num_workers=num_workers,
                       pre_processing=pre_process,
                       verbose_level=verbose_level,
                       min_length=seconds,
                       trim_interval=(0, seconds),
                       **kwargs)        

    if seconds:
        print('[INFO] removing old files from data augmentation')
        # The old files from data augmentation or post processing will be 
        # deleted. Note that if no trimming is performed the audio files will
        # have the same length as the original files, and will not be removed
        # by this code.
        for file in (tqdm(file_list) if int(verbose_level) < 2 else file_list):
            if os.path.isfile(file):
                if int(verbose_level) == 2:
                    print('[INFO] removing', file)
                # os.remove(file)


def process_trim_parallel(audio, output_dir, seconds, trimming_window,
                          verbose_level, num_workers, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_workers)) \
            as executor:
        audio_length = float(get_audio_info(audio, 'duration', verbose_level))
        futures = [executor.submit(pre_process,
                                   output_dir=output_dir,
                                   file_path=audio,
                                   verbose_level=verbose_level,
                                   min_length=int(seconds),
                                   trim_interval=(i, i + seconds),
                                   **kwargs)
                   for i in range(0, int(audio_length) - trimming_window,
                                  trimming_window)]

        kw = {
            'total': len(futures),
            'unit': 'trims',
            'unit_scale': True,
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kw):
            pass


if __name__ == '__main__':
    import json
    import argparse
    import importlib
    from collections import defaultdict
    from random import shuffle
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
                        type=float, default=None)
    parser.add_argument('--rate', help='Set the output rate of audio files.')
    parser.add_argument('--workers', help='Define how many process to run in '
                                          'parallel.', default=4, type=int)
    parser.add_argument('--check', help='Check output directories ignoring '
                                        'files already processed. '
                                        'Note: augmented data will not be '
                                        'checked.',
                        action='store_true')
    parser.add_argument('--just_check', help='Check output directories. A csv '
                                             'log file will be created '
                                             'containing the remaining files to'
                                             ' process. No processing will be '
                                             'performed. '
                                             'Note: augmented data will not be '
                                             'checked.',
                        action='store_true')
    parser.add_argument('--limit', help='Set a limit of files to process. '
                                        'Useful when the data is unbalanced '
                                        'and the process is slow. Set this '
                                        'parameter as -1 to do it '
                                        'automatically. This limit is '
                                        'applied before the pre processing, so '
                                        'it does not balances the data.',
                        type=int)
    parser.add_argument('--length_checking', help='Will force length checking '
                                                  'of each audio. This will '
                                                  'pass --ignore_length to '
                                                  'sox. Can slow down the '
                                                  'process. Recommended if the '
                                                  'processing is skipping some '
                                                  'audio files. ',
                        action='store_true')
    group = parser.add_argument_group('Data augmentation options')
    group.add_argument('--sw', help='Sliding window: augments data '
                                    'by trimming audio files in '
                                    'different positions. Must provide the '
                                    '[seconds] argument',
                       action='store_true')
    group.add_argument('--tw', help='Trimming window:Trims audio files in '
                                    'different positions. Use this if your '
                                    'dataset have large speeches to process. '
                                    'This operation will trim the original file'
                                    ' in small speeches of length [seconds]. '
                                    'Must provide the [seconds] argument.',
                       action='store_true')
    group.add_argument('--sp', help='Speed: augments data '
                                    'by changing the speed of the audio '
                                    'files.',
                       action='store_true')
    group.add_argument('--ns', help='Noise: augments data '
                                    'by adding noise in the audio '
                                    'files.',
                       action='store_true')
    group.add_argument('--pt', help='Pitch: augment data by changing the '
                                    'pitch of audio files.',
                       action='store_true')
    parser.add_argument('--v', help='Change verbosity level (sox output)',
                        default=0)

    # Set arguments settings:
    arguments = parser.parse_args()
    data_dir = arguments.corpus
    output = arguments.output
    seconds = arguments.seconds
    output_rate = arguments.rate
    workers = arguments.workers
    verbose = arguments.v
    limit = arguments.limit
    trim_silence = arguments.trim_silence
    length_checking = arguments.length_checking

    data_augmentation = arguments.pt or arguments.sp or arguments.ns or \
        arguments.sw or arguments.tw

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
            files_paths = files_to_process(all_files_path, out_dir)
        else:
            files_paths = None
        # Set the new base:
        new_base_name = bases_json[base]['lang']

        if log_remaining is not None and files_paths is not None:
            for f in files_paths:
                log_remaining.write('\n' + new_base_name + ',' + base + ',' + f)
        if files_paths is None:
            files_paths = all_files_path

        # Append file paths to the respective language base
        files_list_lang[new_base_name] += files_paths

    if limit is not None:
        print('[INFO] limiting the amount of files to process')
        if limit == -1:
            count = defaultdict(lambda: 0)
            for base in files_list_lang:
                count[base] += len(files_list_lang[base])
            limit = min(count.values())
        for base in files_list_lang:
            shuffle(files_list_lang[base])
            files_list_lang[base] = files_list_lang[base][:limit]

    print('TOTAL FILES TO BE PROCESSED: %d\n' %
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
        create_dataset(
            dataset_dir=output + os.sep + base,
            file_list=files_list_lang[base],
            num_workers=workers,
            pre_processing=pre_process,
            verbose_level=verbose,
            min_length=seconds if seconds is not None else 0,
            rate=output_rate,
            normalize=True,
            remix_channels=True,
            ignore_length=length_checking,
            trim_silence_threshold=trim_silence,
            trim_interval=(0, seconds)  # Disable trim if data augmentation is
            # enabled:
            if seconds is not None and not data_augmentation else None)
        if data_augmentation:
            # The raw files will be removed, that is, all wav files in the
            # output folder, except the processed files.
            raw_files = []
            for dr in os.listdir(output + os.sep + base):
                if dr[0] == '_':
                    raw_files += glob.glob(output + os.sep + base + os.sep + dr
                                           + '*/**/*.wav', recursive=True)
            print('[INFO] processing data augmentation')
            augment_data(output + os.sep + base,
                         file_list=raw_files,
                         sliding_window=2 if arguments.sw else None,
                         trimming_window=seconds if arguments.tw else None,
                         seconds=seconds,
                         noises=list(NOISES) if arguments.ns else None,
                         semitones=list(SEMITONES) if arguments.pt else None,
                         speeds=list(SPEEDS) if arguments.sp else None,
                         num_workers=workers,
                         remix_channels=False,
                         normalize=False,
                         verbose_level=verbose)
            # Remove raw files
            # If the data augmentation is enabled, the length of each audio
            # file may vary because no trimming was performed before.
            # To maintain a standardized dataset it is necessary to remove the
            # old files.
            print('[INFO] removing old files')
            for fp in (tqdm(raw_files) if int(verbose) < 2 else raw_files):
                if os.path.isfile(fp):
                    if int(verbose) == 2:
                        print('[INFO] removing', fp)
                    os.remove(fp)
        for dr in os.listdir(output + os.sep + base):
            if len(os.listdir(output + os.sep + base + os.sep + dr)) == 0:
                os.rmdir(output + os.sep + base + os.sep + dr)
            elif dr[0] == '_':
                os.rename(output + os.sep + base + os.sep +
                          dr, output + os.sep + base + os.sep + dr[1:])
