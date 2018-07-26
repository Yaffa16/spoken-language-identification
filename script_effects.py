# This script applies effects to audio files

# Script imports:
import os
import argparse
from preprocessing.effects import *
from util.audiosignal import AudioSignal
from util.timing import Timer


if __name__ == '__main__':
    # Command line arguments:
    parser = argparse.ArgumentParser(description='Audio effects')
    parser.add_argument('--source', help='Source. If it is a file, the effect '
                                         'will be applied to the file, if is a '
                                         'directory, all the files in the '
                                         'source will be processed. Default to '
                                         'current working directory.')
    parser.add_argument('--output', help='Output directory. If the directory '
                                         'not exists, the program will be '
                                         'create them automatically. Default '
                                         'to current working directory.')
    parser.add_argument('effect', help='Effect to apply.',
                        choices=AVAILABLE_EFFECTS)
    parser.add_argument('--effect_options', help='Effect options arguments',
                        nargs='*')
    parser.add_argument('--d', help='Show detailed information (documentation) '
                                    'and exit.', action='store_true')
    # TODO: record option

    args = parser.parse_args()
    if args.d:
        for effect in Effect.__subclasses__():
            print(effect.help())
        exit(0)

    if args.source is None:
        file_paths = os.listdir('./')
    elif os.path.isfile(args.source):
        file_paths = [args.source]
    else:
        file_paths = os.listdir(args.source)

    if args.effect_options is not None:
        kwargs = dict(args.effect_options[i:i+2]
                      for i in range(0, len(args.effect_options), 2))

    with Timer() as timer:
        for f_name in file_paths:
            if '.wav' not in f_name:
                continue
            print('Processing ', f_name)
            audio = AudioSignal(f_name)
            audio.load()
            effect = AVAILABLE_EFFECTS[args.effect](audio)
            with Timer() as t:
                if args.effect_options is not None:
                    effect.apply(**kwargs)
                else:
                    effect.apply()
                if args.output:
                    effect.write_audio(args.output)
                else:
                    effect.write_audio()
            print('Time taken: %.03f sec.' % t.interval)
    print('Total time: %.03f sec.' % timer.interval)
