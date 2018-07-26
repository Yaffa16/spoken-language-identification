# Adapted | spoken-language-identification
# Not working yet

from preprocessing.spectrogram import spectrogram
from util import convert
from util import timing
import numpy as np
import inspect
import argparse
import os


def augment_wav_data(handler: callable, data_path: str, spec_output_dir: str,
                     spec_output_name: str, **kargs):
    """
    Augment data and generate spectrogram of wav files.

    :param data_path: path of the input wav file
    :param spec_output_dir: output directory of the spectrogram
    :param spec_output_name: output name for the spectrogram
    :param handler: Callable object witch handles and generates spectrogram of
                    wav files.
    :param kargs: Additional kwargs are passed on to the handler function
                  object.
    """
    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations \
        = inspect.getfullargspec(handler)
    if ('audiopath', 'plotpath', 'name') not in args:
        raise ValueError('The handler callable object do not accept '
                         'expected args. Expected callable object: '
                         '\n{}'.format(inspect.getdoc(spectrogram)))

    # for augment_idx in range(0, 20):
    #     alpha = np.random.uniform(0.9, 1.1)
    #     offset = np.random.randint(90)
    #     handler(audiopath=data_path, plotpath)
    #
    #     handler(audiopath=data_path, channel=0, plotpath=spec_output_dir, name=spec_output_name + '.' +
    #              str(augment_idx) + '.png', alpha=alpha, offset=offset)
    # TODO


@timing
def augment_data(data_dir: str, specs_dir: str, file_format: str='wav',
                 file_names_path: str=None, wav_dir: str=None):
    """
    Augment data and generate spectrogram of audio files.

    :param data_dir: source directory of audio files
    :param specs_dir: output directory
    :param file_format: format of source audio files (default=wav)
    :param file_names_path: path for file containing file names to process
        In case no file is provided, all the files of the working source
        directory will be considered (default=None)
    :param wav_dir: output for converted audio files in wav format
        In case no path is provided, a temp file will be generated and removed
        in sequence (default=None)
    """
    if file_names_path is None:
        file_names = os.listdir(data_dir)
    else:
        file_names = open(file_names_path, 'r').readlines()
    # for i, line in enumerate(file_names_path.readlines()[
    #                          1:]):
    for i, line in enumerate(file_names):
        file_path = line.split(',')[0]
        file_name = str(file_path[:-4])
        if wav_dir is None:
            wav_file_name = 'tmp.wav'
        else:
            wav_file_name = file_name + '.wav'

        try:
            if file_format != 'wav':
                convert(file_path, wav_dir + '/' + wav_file_name)

            # augment_wav_data(wav_dir + '/' + wav_file_name, specs_dir,
            #                  file_name)
            # TODO
        except RuntimeError:
            print('WARNING: file {} not converted, skipping'.format(file_path))

        if wav_dir is None:
            os.remove(wav_file_name)
        print("processed %d files" % (i + 1))


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Augment data and generate '
                                                 'spectrogram of audio files.')
    parser.add_argument('-data_dir', help='Source directory of the audio '
                                          'files.')
    parser.add_argument('-f', help='Format of audio files. In case the format '
                                   'is not wav, each file will be converted '
                                   'to wav.')
    parser.add_argument('-specs_dir', help='Output directory of generated '
                                           'spectrogram files.')
    parser.add_argument('--file_names', help='Source which contains audio '
                                             'file names to process. If no '
                                             'file is provided, all files in '
                                             'the working folder will be '
                                             'processed.')
    parser.add_argument('--wav_dir', help='Directory to save converted wav '
                                          'files. If no directory is provided, '
                                          'the converted files will be '
                                          'deleted')
    parser.add_argument('-handler', help='Function or software which '
                                          'will generate the spectrogram. '
                                          'Default to internal script.',
                        default='stft')
    args = parser.parse_args()

    augment_data(args.data_dir, args.specs_dir, args.f, args.file_names,
                 args.wav_dir)
