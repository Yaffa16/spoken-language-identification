"""
This script records and tests in real time the language being spoken.
"""

import pyaudio
import wave
import numpy as np
import librosa
import os
import sys
from keras.models import load_model
from keras.models import Model
from preprocessing.spectrogram import *
from multiprocessing import Process
from util.imgload import img_load
from pydub import AudioSegment
import subprocess

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = 'temp.wav'


def loader(p):
    y, sr = librosa.load(p, duration=5)
    d = librosa.feature.mfcc(y, sr, n_mfcc=13)
    return d.reshape(d.shape[0], d.shape[1], 1)  # shape?


def real_time(model_path: str):
    """
    Tests in real time a model.

    :param model_path: str
        Path to the saved model.
    """
    print('> Loading model')
    model = load_model(model_path)
    print('Done')

    from script_create_dataset import trim_silence_audio, remix, convert_rate, \
        trim

    while True:
        print('Gravando...')
        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

        temp = set()
        temp.add(remix(WAVE_OUTPUT_FILENAME, '.', 'temp1'))
        temp.add(remix('temp1.wav', '.', 'temp2'))
        temp.add(convert_rate(16000, 'temp2.wav', '.', 'temp3'))
        data = loader('temp3.wav')
        p = model.predict(np.asarray([data]))

        print('> Predição:')
        print(p)
        os.system('play {} -t waveaudio'.format('temp3.wav'))
        for a in temp:
            os.remove(a)


if __name__ == '__main__':
    # todo: add command line arguments
    p = Process(target=real_time,
                kwargs={'model_path': sys.argv[1]})
    p.start()
    input('Press any key to stop\n')
    p.terminate()
    os.remove(WAVE_OUTPUT_FILENAME)
