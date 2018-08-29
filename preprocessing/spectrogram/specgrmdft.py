# Adapted | spoken-language-identification

# todo: document function params
# todo: check module, rewrite if necessary

import os
import numpy as np
import scipy.io.wavfile as wav
import PIL.Image as Image
from numpy.lib import stride_tricks


def stft(sig, frame_size, overlap_fac=0.5, window=np.hanning):
    """Short time fourier transform of audio signal"""
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap_fac * frame_size))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    b = int(np.floor(frame_size / 2.0))
    # c= 512
    a = np.zeros(b)
    samples = np.append(a, sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frame_size) / float(hop_size)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))

    frames = stride_tricks.as_strided(samples, shape=(cols, frame_size),
                                      strides=(samples.strides[0] * hop_size,
                                               samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, alpha=1.0, f0=0.9, fmax=1):
    """Scale frequency axis logarithmically"""
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)

    scale = map(lambda x: x * alpha if x <= f0 else \
        (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0, scale)
    scale = np.fromiter(scale, dtype=np.float64)
    scale *= (freqbins - 1) / max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if i < 1 or i + 1 >= freqbins:
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if totw[i] > 1e-6:
            freqs[i] /= totw[i]

    return newspec, freqs


def plotstft(audiopath: str, plotpath: str = None, name: str = 'tmp',
             binsize=2 ** 10, alpha=1):
    """Plot spectrogram using a short time fourier transform"""
    samplerate, samples = wav.read(audiopath)
    if len(samples.shape) > 1:  # Code to prevent 'too many indices for array'
                                # exception.
        samples = samples[:, 0]
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)

    # ims = ims[0:256, offset:offset + 768]  # 0-11khz, ~9s interval
    ims = ims[0:256, 0:0 + 768]  # 0-11khz, ~9s interval
    ims = ims[0:256, :]  # 0-11khz, ~10s interval

    image = Image.fromarray(ims)
    image = image.convert('L')
    if plotpath is not None:
        image.save(plotpath + '/' + name + '.png')
    else:
        image.save(name + '.png')


def specgram_dft(audiopath: str, plotpath: str=None, name: str=None, **kwargs):
    """
    Generate a spectrogram of an audio file using specgrmstft.plotstft function.

    The output will be in png format.

    :param audiopath: string
        Path of the audio file.
    :param plotpath: string
        Path to plot the spectrogram. Default to the current working directory.
    :param name: string
        Name of the output image.
    :param kwargs:
        Additional kwargs are passed on to specgrmstft.plotstft function.

    :Example:
    # >>> from
    """
    if name is None:
        name = audiopath.split(os.sep)[-1]

    if plotpath is not None and not os.path.isdir(plotpath):
        os.makedirs(plotpath)
    plotstft(audiopath=audiopath, plotpath=plotpath, name=name, **kwargs)
