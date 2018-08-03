from scipy.io import wavfile
from core.exceptions import NotLoadedError
import numpy as np
import pyaudio
import wave


class AudioSignal:
    """Represents an audio signal."""

    # Default options for recording operations using PyAudio
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024

    def __init__(self, audiopath: str):
        """
        Constructs a new audio signal.

        :param audiopath: str
            Path to the audio file.
        """
        self._audiopath = audiopath
        self._fs = None
        self._data = None
        self._file = None
        self._fs = None
        self._data = None

    @property
    def fs(self) -> int:
        """Returns the sample rate."""
        if self._fs is None:
            raise NotLoadedError('Must load or record an audio before access '
                                 'an audio property.')
        return self._fs

    @property
    def data(self):
        """Returns the data."""
        if self._data is None:
            raise NotLoadedError('Must load or record an audio before access '
                                 'an audio property.')
        return self._data

    @property
    def name(self) -> str:
        """Returns the name of the audio file."""
        return self._audiopath.split('/')[-1]

    @property
    def dir(self) -> str:
        """Returns the directory of the file."""
        return '/'.join(self._audiopath.split('/')[:-1])

    @fs.setter
    def fs(self, value: int):
        """Sets the sample rate with a new value."""
        if self._fs is None:
            raise NotLoadedError('Must load or record an audio before access '
                                 'an audio property.')
        self._fs = value

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data with a new value."""
        if self._data is None:
            raise NotLoadedError('Must load or record an audio before access '
                                 'an audio property.')
        self._data = value

    def load(self):
        """Reads a wav file"""
        self._fs, self._data = wavfile.read(self._audiopath)

    def record(self, seconds: int=10, **kargs):
        """
        Records a new audio and store it in the provided path.

        :param seconds: int
            Time in seconds to record the audio.
        :param kargs:
            Additional kargs are passed on to the open method of a
            pyaudio.PyAudio object.
        """
        audio = pyaudio.PyAudio()
        kargs.setdefault('format', pyaudio.paInt16)
        kargs.setdefault('channels', AudioSignal.CHANNELS)
        kargs.setdefault('rate', AudioSignal.RATE)
        kargs.setdefault('input', True)
        kargs.setdefault('frames_per_buffer', AudioSignal.CHUNK)
        stream = audio.open(**kargs)

        frames = []
        for i in range(0, int(kargs.get('rate') /
                              kargs.get('frames_per_buffer') * seconds)):
            data = stream.read(kargs.get('frames_per_buffer'))
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = wave.open(self._audiopath, 'wb')
        wave_file.setnchannels(kargs.get('channels'))
        wave_file.setsampwidth(audio.get_sample_size(kargs.get('format')))
        wave_file.setframerate(kargs.get('rate'))
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
