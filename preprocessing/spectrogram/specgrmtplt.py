import os
import wave
import numpy as np
import matplotlib.pyplot as plt


def specgram_mtplt(audiopath: str, plotpath: str=None, name: str='tmp',
                   **kwargs):
    """
    Generate a spectrogram of an audio file using matplotlib.

    The output will be in png format.

    :param audiopath: string
        Path of the audio file.
    :param plotpath: string
        Path to plot the spectrogram. Default to the current working directory.
    :param name: string
        Name of the output image.
    :param kwargs:
        Additional kwargs are passed on to matplotlib.pyplot.specgram function.
    """
    if plotpath is not None and not os.path.isdir(plotpath):
        os.makedirs(plotpath)
    kwargs.setdefault('NFFT', 2 ** 10)
    kwargs.setdefault('cmap', 'gray_r')

    wf = wave.open(audiopath, 'rb')
    fs = wf.getframerate()
    N = wf.getnframes()
    duration = N / float(fs)
    bytes_per_sample = wf.getsampwidth()
    bits_per_sample = bytes_per_sample * 8
    dtype = 'int{0}'.format(bits_per_sample)
    channels = wf.getnchannels()

    audio = np.fromstring(
        wf.readframes(int(duration * fs * bytes_per_sample / channels)),
        dtype=dtype)
    audio.shape = (audio.shape[0] // channels, channels)
    fig = plt.figure(figsize=(10, 4))

    plt.specgram(audio[:, 0], Fs=fs, noverlap=0, **kwargs)
    plt.axis(ymin=0, ymax=12000)  # Note: trying to limit frequency range
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if plotpath is not None:
        plt.savefig(plotpath + '/' + name + '.png')
    else:
        plt.savefig(name + '.png')
    plt.close()
