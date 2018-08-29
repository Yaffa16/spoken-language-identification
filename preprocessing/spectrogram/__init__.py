from preprocessing.spectrogram.specgrmtplt import specgram_mtplt
from preprocessing.spectrogram.specgrmsox import specgram_sox
from preprocessing.spectrogram.specgrmlbrs import specgram_lbrs
from preprocessing.spectrogram.specgrmdft import specgram_dft

DEFAULT_HANDLER_SPEC = specgram_lbrs


def _spectrogram(audiopath: str, plotpath: str, name: str, **kargs):
    """
    Must generate a spectrogram of an audio file. Use this signature to develop
    custom spectrogram handler functions.

    Expected output: png.

    :param audiopath: Path of the audio file.
    :param plotpath: Path to plot the spectrogram.
    :param name: Name of the output image.
    :param kargs: Additional kwargs are passed on to the handler object.
    """
    raise NotImplementedError('Must implemented spectrogram function')


def spectrogram(audiopath, plotpath, name, **kwargs):
    """
    Generate a spectrogram of an audio file using the default callable object.

    The output will be in png format.

    :param audiopath:
        Path of the audio file.
    :param plotpath:
        Path to plot the spectrogram.
    :param name:
        Name of the output image.
    :param kwargs:
        Additional kwargs are passed on to the handler callable object.
    """
    DEFAULT_HANDLER_SPEC(audiopath=audiopath, plotpath=plotpath, name=name,
                         kargs=kwargs)
