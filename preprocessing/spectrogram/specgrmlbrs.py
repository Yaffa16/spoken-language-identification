import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow

FIG_SIZE = None  # Default matplotlib.figure fig size handling


def specgram_lbrs(audiopath: str, plotpath: str=None, name: str='tmp',
                  cmap: str='gray_r', algorithm='default', y_axis=None,
                  **kwargs):
    """
    Generates a spectrogram of an audio file using librosa.display.specshow
    function.

    The output will be in png format.

    :param audiopath: string
        Path of the audio file.
    :param plotpath: string
        Path to plot the spectrogram. Default to the current working directory.
    :param name: string
        Name of the output image.
    :param cmap: string
        Automatic colormap detection
        See matplotlib.pyplot.pcolormesh.
    :type algorithm: str or callable
    :param algorithm: Algorithm to use to compute the spectrogram.
        Available algorithms: 'default', 'mel', 'log'.
        The 'default' mode uses the librosa.stft function to compute the
        spectrogram data, the 'mel' uses the librosa.feature.melspectrogram
        function and 'log' uses librosa.cqt function.
        Expected return type of the algorithm: np.ndarray [shape=(Any, t)]
    :param y_axis: None or str.
        Range for the y-axes. This parameter is passed to the
        librosa.display.specshow function.
    :param kwargs:
        Additional kwargs are passed on to the defined algorithm function.

    """
    if plotpath is not None and not os.path.isdir(plotpath):
        os.makedirs(plotpath)
    if algorithm not in ['mel', 'log', 'default'] or not callable:
        raise ValueError('Unrecognized scale or not a callable object.')

    # Load audio and convert it to mono
    y, sr = librosa.load(audiopath)
    y = librosa.core.to_mono(y)

    # Apply algorithm to obtain an array of spectrogram data
    if algorithm == 'default':
        spec_data = librosa.stft(y, **kwargs)
        # Convert the data spectrogram to decibel units
        spec_data = librosa.power_to_db(librosa.magphase(spec_data, power=2)[0],
                                        ref=np.max)
    elif algorithm == 'mel':
        kwargs.setdefault('n_mels', 128)
        kwargs.setdefault('fmax', 8000)
        spec_data = librosa.feature.melspectrogram(y=y, sr=sr, **kwargs)
        # Convert the data spectrogram to decibel units
        spec_data = librosa.power_to_db(spec_data, ref=np.max)
    elif algorithm == 'log':
        spec_data = librosa.cqt(y, sr, **kwargs)
        # Convert the data spectrogram to decibel units
        spec_data = librosa.power_to_db(librosa.magphase(spec_data, power=2)[0],
                                        ref=np.max)
    else:
        spec_data = algorithm(y=y, sr=sr, **kwargs)

    # Plot spectrogram
    fig = plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(spec_data, sr=sr, cmap=cmap, y_axis=y_axis)
    plt.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if plotpath is not None:
        plt.savefig(plotpath + '/' + name + '.png')
    else:
        plt.savefig(name + '.png')
    plt.close()
