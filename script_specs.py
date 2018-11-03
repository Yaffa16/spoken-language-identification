"""
This script generates spectrogram of wav audio files using parallel processing.
"""
# todo: ignore large files
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
        if file_name + '.png' in output_files:
            remaining_files.remove(el)
            counter += 1
    print('There are {} files remaining to process, '
          'and {} spectrogram images in {}.'.format(len(files_list)
                                                    - counter,
                                                    counter,
                                                    output_dir))
    return remaining_files


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
            log.write('\nExceptions for {} call at {}'.format(fun.__name__,
                                                              time.asctime()))
            for f in futures:
                if f.exception() is not None:
                    log.write('\n' + str(f.exception()))


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
    parser.add_argument('--dft', help='Process files using the default '
                                      'algorithm.', action='store_true',
                        default=True)
    parser.add_argument('--sox', help='Process files using sox algorithm.',
                        action='store_true', default=False)
    parser.add_argument('--librosa_default', help='Process files using '
                                                  'the librosa algorithm '
                                                  '(default arguments)',
                        action='store_true', default=False)
    parser.add_argument('--librosa_log', help='Process files using '
                                              'the librosa algorithm'
                                              '(log)',
                        action='store_true', default=False)
    parser.add_argument('--librosa_mel', help='Process files using '
                                              'the librosa algorithm'
                                              '(mel)',
                        action='store_true', default=False)
    parser.add_argument('--librosa_dft_log', help='Process files using '
                                              'the librosa algorithm'
                                              '(y_axis=log)',
                        action='store_true', default=False)
    parser.add_argument('--librosa_dft_mel', help='Process files using '
                                                  'the librosa algorithm'
                                                  '(y_axis=mel)',
                        action='store_true', default=False)
    parser.add_argument('--matplotlib', help='Process files using matplotlib.',
                        action='store_true', default=False)
    parser.add_argument('--v', help='Change verbosity level (sox output)',
                        default=0)
    os.makedirs('logs/scripts/', exist_ok=True)

    # Set source and output directories:
    arguments = parser.parse_args()
    data_dir = arguments.source
    output = arguments.output
    dft = arguments.dft
    sox = arguments.sox
    librosa_default = arguments.librosa_default
    librosa_log = arguments.librosa_log
    librosa_mel = arguments.librosa_mel
    librosa_dft_log = arguments.librosa_dft_log
    librosa_dft_mel = arguments.librosa_dft_mel
    matplotlib = arguments.matplotlib

    # Make directories
    if dft:
        os.makedirs(output + '/dft', exist_ok=True)
    if sox:
        os.makedirs(output + '/sox', exist_ok=True)
    if librosa_default:
        os.makedirs(output + '/librosa/default', exist_ok=True)
    if librosa_log:
        os.makedirs(output + '/librosa/log', exist_ok=True)
    if librosa_mel:
        os.makedirs(output + '/librosa/mel', exist_ok=True)
    if librosa_dft_log:
        os.makedirs(output + '/librosa/default/log', exist_ok=True)
    if librosa_dft_mel:
        os.makedirs(output + '/librosa/default/mel', exist_ok=True)
    if matplotlib:
        os.makedirs(output + '/matplotlib', exist_ok=True)

    # files = glob.glob(data_dir + os.sep + '**/*.wav', recursive=True)
    # files = [f.split(os.sep)[-1] for f in files]
    files = os.listdir(data_dir)
    files_dft = []
    files_sox = []
    files_lbr_dft = []
    files_lbr_log = []
    files_lbr_mel = []
    files_lbr_dl = []
    files_lbr_dm = []
    files_plt = []

    # Check processed files to ignore them
    if arguments.check:
        print("\n> CHECKING")
        if dft:
            files_dft = files_to_process(files, output + '/dft')
        if sox:
            files_sox = files_to_process(files, output + '/sox')
        if librosa_default:
            files_lbr_dft = files_to_process(files, output + '/librosa/default')
        if librosa_log:
            files_lbr_log = files_to_process(files, output + '/librosa/log')
        if librosa_mel:
            files_lbr_mel = files_to_process(files, output + '/librosa/mel')
        if librosa_dft_log:
            files_lbr_dl = files_to_process(files, output +
                                            '/librosa/default/log')
        if librosa_dft_mel:
            files_lbr_dm = files_to_process(files, output +
                                            '/librosa/default/mel')
        if matplotlib:
            files_plt = files_to_process(files, output + '/matplotlib')
    else:
        if dft:
            files_dft = files
        if sox:
            files_sox = files
        if librosa_default:
            files_lbr_dft = files
        if librosa_log:
            files_lbr_log = files
        if librosa_mel:
            files_lbr_mel = files
        if librosa_dft_log:
            files_lbr_dl = files
        if librosa_dft_mel:
            files_lbr_dm = files
        if matplotlib:
            files_plt = files

    print('\n> PROCESSING')
    with Timer() as timer:
        if dft:
            process_files_in_parallel(fun=specgram_dft, files_list=files_dft,
                                      data_path=data_dir,
                                      plot_path=output + '/dft',
                                      workers=arguments.workers)
        if sox:
            process_files_in_parallel(fun=specgram_sox, files_list=files_sox,
                                      data_path=data_dir,
                                      plot_path=output + '/sox',
                                      workers=arguments.workers)
        if librosa_default:
            process_files_in_parallel(fun=specgram_lbrs,
                                      files_list=files_lbr_dft,
                                      data_path=data_dir,
                                      plot_path=output + '/librosa/default',
                                      workers=arguments.workers)
        if librosa_log:
            process_files_in_parallel(fun=specgram_lbrs,
                                      files_list=files_lbr_log,
                                      data_path=data_dir, plot_path=output +
                                      '/librosa/log', algorithm='log',
                                      workers=arguments.workers)
        if librosa_mel:
            process_files_in_parallel(fun=specgram_lbrs,
                                      files_list=files_lbr_mel,
                                      data_path=data_dir, plot_path=output +
                                      '/librosa/mel', algorithm='mel',
                                      workers=arguments.workers)
        if librosa_dft_log:
            process_files_in_parallel(fun=specgram_lbrs,
                                      files_list=files_lbr_dl,
                                      data_path=data_dir, plot_path=output +
                                      '/librosa/default/log', y_axis='log',
                                      workers=arguments.workers)
        if librosa_dft_mel:
            process_files_in_parallel(fun=specgram_lbrs,
                                      files_list=files_lbr_dm,
                                      data_path=data_dir, plot_path=output +
                                      '/librosa/default/mel', y_axis='mel',
                                      workers=arguments.workers)
        if matplotlib:
            process_files_in_parallel(fun=specgram_mtplt, files_list=files_plt,
                                      data_path=data_dir,
                                      plot_path=output + '/matplotlib',
                                      workers=arguments.workers)

    print('WORK DONE, total taken time: %.03f sec.' % timer.interval)

    if arguments.check_after:
        print("\n> CHECKING")
        if dft:
            files_dft = files_to_process(files, output + '/dft')
        if sox:
            files_sox = files_to_process(files, output + '/sox')
        if librosa_default:
            files_lbr_dft = files_to_process(files, output +
                                             '/librosa/default')
        if librosa_log:
            files_lbr_log = files_to_process(files, output + '/librosa/log')
        if librosa_mel:
            files_lbr_mel = files_to_process(files, output + '/librosa/mel')
        if librosa_dft_log:
            files_lbr_dm = files_to_process(files, output +
                                            '/librosa/default/log')
        if librosa_dft_mel:
            files_lbr_dl = files_to_process(files, output +
                                            '/librosa/default/mel')
        if matplotlib:
            files_plt = files_to_process(files, output + '/matplotlib')

        with open('logs/scripts/specs_remaining_files.txt', 'w') as file:
            for rem_file in list(files_dft + files_sox + files_lbr_dft +
                                 files_lbr_dl + files_lbr_log + files_lbr_mel +
                                 files_lbr_dm + files_plt):
                file.write('\n' + rem_file)
