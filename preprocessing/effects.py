"""
This module implements some audio effects generators.
"""
from util.audiosignal import AudioSignal
from util.timing import timing
from abc import abstractmethod
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import warnings


class Effect:
    """Represents an audio effect."""

    def __init__(self, signal: AudioSignal):
        """
        Must construct a new applicable effect to an audio signal.

        :param signal: AudioSignal
            An audio signal to apply the effect.
        """
        self._signal = signal

    @abstractmethod
    def apply(self, *args, **kwargs):
        """
        Must apply the implemented effect on the signal.

        :param args:
            Additional args are passed on to the implemented effect.
        :param kwargs:
            Additional kwargs are passed on to the function which generates the
            noise.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_audio(self, path: str='', name: str=None, *args, **kwargs):
        """
        Must write the signal in a file.

        :param name: str
            Name of the output file. Default could be the signal name.
        :param path: str
            Path of the output file. Default could be the path of the
            AudioSignal. Default to nothing, that is, the audio will be saved
            based on the behavior of the handler function, usually, the
            current working folder.
        :param args:
            Additional args could be passed on to the implemented write
            function.
        :param kwargs:
            Additional kwargs could be passed on to the internal handle which
            writes the file.
        """
        raise NotImplementedError()

    @property
    def signal(self):
        """Returns the respective signal of this effect"""
        return self._signal

    @signal.setter
    def signal(self, signal_data):
        self._signal.data = signal_data

    @classmethod
    @abstractmethod
    def help(cls):
        raise NotImplementedError()


class Noise(Effect):
    """Represents an applicable noise effect to an signal"""

    available_noises = ['white']

    def __init__(self, signal: AudioSignal):
        """
        Constructs a new applicable effect to an audio signal.

        :param signal: AudioSignal
            An audio signal to apply the effect.
        """
        super(Noise, self).__init__(signal)

    @staticmethod
    def white_noise(shape: tuple, factor: float=0.1) -> np.ndarray:
        """
        Generates a random white_noise

        :param shape: tuple (len=2)
            The shape of the generated noise.
        :param factor:
            A factor to increase or decrease the amplitude of the generated
            noise.
        :return: numpy.ndarray shape=(shape)
            Returns the generated noise with the provided parameters.
        """
        if not len(shape) == 2:
            raise ValueError('Not acceptable shape. Must have length of 2.')

        noise = np.random.uniform(-1, 1, shape[0]*shape[1])
        scaled = np.int16(noise / np.max(np.abs(noise)) * 32767 * float(factor))
        return np.reshape(scaled, shape)

    def apply(self, noise: str='white', **kwargs):
        """
        Applies a noise effect on the signal.

        :param noise: str or array with noise to apply to.
            Type of noise. Default to 'white' to generate a white noise.
            See Noise.available_noises to see a list of the implemented
            noises.
        :param kwargs:
            Additional kwargs are passed on to the noise generator function.
        """
        if noise not in self.available_noises or not hasattr(noise, '__len__'):
            raise ValueError('Unrecognized or not applicable noise: {}'
                             .format(noise))

        if noise == 'white':
            data = np.copy(self.signal.data)
            data += Noise.white_noise(shape=self.signal.data.shape, **kwargs)
            self.signal.data = data
        elif hasattr(noise, '__len__'):
            if len(noise) != len(self.signal.data):
                warnings.warn('Noise length and signal length are not equal, '
                              'provide an equal length noise signal to '
                              'suppress this warning.')
            data = np.copy(self.signal.data)
            data += noise
            self.signal.data = data

    def write_audio(self, path: str='', name: str=None, *args, **kwargs):
        """
        Writes the signal in a file.

        :param name: str
            Name of the output file. Default to the signal name.
        :param path: str
            Path of the output file. Default to nothing, the file will be saved
            in the current working folder.
        :param kwargs:
            Additional kwargs are passed on to scipy.io.wavfile.write function.
        """
        if name is None:
            name = self.signal.name

        kwargs.setdefault('rate', 44100)
        wavfile.write(filename=path + 'noised-' + name, data=self.signal.data,
                      **kwargs)

    @classmethod
    def help(cls):
        return __class__.__name__ + ' effect: ' + str(cls.__doc__) + \
               '.\nInstructions: ' + str(cls.apply.__doc__)

    def __str__(self):
        return self.__class__.__name__


class Normalize(Effect):
    """Represents an applicable normalize effect to an signal"""

    def __init__(self, signal: AudioSignal):
        """
        Constructs a new applicable effect to an audio signal.

        :param signal: AudioSignal
            An audio signal to apply the effect.
        """
        super(Normalize, self).__init__(signal)
        self._song = AudioSegment.from_file(self.signal.dir + '/' +
                                            self.signal.name)
        self._normalized = None

    @staticmethod
    def _match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def apply(self, **kwargs):
        """
        Applies a normalize effect on the signal.

        :param kwargs:
            Additional kwargs are ignored.
        """
        self._normalized = Normalize._match_target_amplitude(self._song, -20.0)

    def write_audio(self, path: str = '', name: str = None, *args, **kwargs):
        """
        Writes the signal in a file.

        :param name: str
            Name of the output file. Default to the signal name.
        :param path: str
            Path of the output file. Default to nothing, the file will be saved
            in the current working folder.
        :param kwargs:
            Additional kwargs are passed on to AudioSegment.export method.
        """
        if name is None:
            name = self.signal.name + '-normalized'

        kwargs.setdefault('format', 'wav')
        self._normalized.export(self.signal.dir + '/normalized-' + name,
                                **kwargs)

    @classmethod
    def help(cls):
        return __class__.__name__ + ' effect: ' + str(cls.__doc__) + \
               '.\nInstructions: ' + str(cls.apply.__doc__)

    def __str__(self):
        return self.__class__.__name__


class Trim(Effect):
    """Represents an applicable trim effect to an signal"""

    def __init__(self, signal: AudioSignal):
        """
        Constructs a new applicable effect to an audio signal.

        :param signal: AudioSignal
            An audio signal to apply the effect.
        """
        super(Trim, self).__init__(signal)
        self._song = AudioSegment.from_file(self.signal.dir + '/' +
                                            self.signal.name)
        self._trimmed = None

    def apply(self, start_time: int=0, end_time: int=None, **kwargs):
        """
        Applies a trim effect on the signal.

        :param start_time: int
            Start time in seconds. Default to 0.
        :param end_time: int
            End time in seconds. Default to None (end of audio).
        :param kwargs:
            Additional kwargs are ignored.
        """
        # Time to miliseconds
        start_time = start_time * 1000
        end_time = end_time * 1000
        if end_time < start_time:
            raise ValueError('End time must be greater than start time')

        if start_time < 0:
            start_time = 0
        if end_time >= len(self._song) or end_time is None:
            end_time = len(self._song) - 1
        self._trimmed = self._song[start_time:end_time]

    def write_audio(self, path: str = '', name: str = None, *args, **kwargs):
        """
        Writes the signal in a file.

        :param name: str
            Name of the output file. Default to the signal name.
        :param path: str
            Path of the output file. Default to nothing, the file will be saved
            in the current working folder.
        :param kwargs:
            Additional kwargs are passed on to AudioSegment.export method.
        """
        if name is None:
            name = self.signal.name + '-trimmed'

        kwargs.setdefault('format', 'wav')
        self._trimmed.export(self.signal.dir + '/trimmed-' + name, **kwargs)

    @classmethod
    def help(cls):
        return __class__.__name__ + ' effect: ' + str(cls.__doc__) + \
               '.\nInstructions: ' + str(cls.apply.__doc__)

    def __str__(self):
        return self.__class__.__name__


AVAILABLE_EFFECTS = {
    Noise.__name__: Noise,
    Trim.__name__: Trim,
    Normalize.__name__: Normalize
}