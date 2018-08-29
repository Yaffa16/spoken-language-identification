"""
specgrmsox: generate spectrogram using sox software

This module uses the "'sox'" software to generate spectrogram of audio files.
Please be sure you have this software installed in your PATH system variable.
More details in http://sox.sourceforge.net
"""

import os


def specgram_sox(audiopath: str, plotpath: str=None, name: str=None,
                 effect_options='-m -r', **kwargs):
    """
    Generate a spectrogram of an audio file using the sox software.

    The output will be in png format.

    :param audiopath: string
        Path of the audio file.
    :param plotpath: string
        Path to plot the spectrogram. Default to the current working directory.
    :param name: string
        Name of the output image.
    :param effect_options: string
        Sox effect options. Default: -m for monochromatic settings and -r to
        suppress the display of axes and legends.
    :param kargs:
        Additional kargs are passed on to global options at sox command line
        arguments. Default to [('remix', '2'), ('rate', '16k')].

    """
    if plotpath is not None and not os.path.isdir(plotpath):
        os.makedirs(plotpath)
    kwargs.setdefault('rate', '16k')
    global_options = ''
    for k in kwargs:
        global_options += ' ' + k + ' ' + kwargs[k]

    if name is None:
        name = audiopath.split('/')[-1]

    if plotpath is None:
        output = name + '.png'
    else:
        output = plotpath + '/' + name + '.png'

    command = 'sox -V0 {} -n {} spectrogram {} -o {}'.format(audiopath,
                                                             global_options,
                                                             effect_options,
                                                             output)

    os.system(command)
    if not os.path.isfile(output):
        raise RuntimeError('Not able to generate spectrogram of file {} '
                           '(The output image does not exists, check either if '
                           'sox is correctly configured in your system or if '
                           'the command line arguments are correct.)'
                           '\n\tWhen trying to execute {}'.
                           format(audiopath, command))
