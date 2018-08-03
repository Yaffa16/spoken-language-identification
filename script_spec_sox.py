# This script generates spectrogram images from wav audios using the
# preprocessing.spectrogram.specgram_sox function.

# Script imports:
import os
import argparse
from util.towav import convert
from preprocessing.spectrogram import specgram_sox


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Generate spectrogram of '
                                                 'audio files.')
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
                                             'the source directory will be '
                                             'processed.')
    parser.add_argument('--wav_dir', help='Directory to save converted wav '
                                          'files. If no directory is provided, '
                                          'the converted files will be '
                                          'deleted')

    args = parser.parse_args()
    data_dir = args.data_dir
    file_names_path = args.file_names
    wav_dir = args.wav_dir
    file_format = args.file_format
    y_axis = args.y_axis
    alg = args.algorithm

    if file_names_path is None:
        file_names = os.listdir(data_dir)
    else:
        file_names = open(file_names_path, 'r').readlines()

    for i, line in enumerate(file_names):
        file_path = str(line.split(',')[0])
        file_name = str(file_path[:-4])
        if wav_dir is None:
            wav_file_name = 'tmp.wav'
        else:
            wav_file_name = file_name + '.wav'

        try:
            if file_format != 'wav':
                convert(file_path, wav_dir + '/' + wav_file_name)

                specgram_sox(audiopath=wav_dir + '/' + wav_file_name,
                             name=wav_file_name)

        except RuntimeError:
            print('WARNING: file {} not converted, skipping'.format(file_path))

        if wav_dir is None:
            os.remove(wav_file_name)
        print("processed %d files of %d" % (i + 1, len(file_names)))
