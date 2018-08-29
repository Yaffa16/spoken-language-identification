"""
This script records and tests in real time the language being spoken.
"""

import pyaudio
import wave
import numpy as np
import os
from keras.models import load_model
from preprocessing.spectrogram import *
from multiprocessing import Process
from util.imgload import img_load

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = 'test.wav'
IMG_OUTPUT_FILENAME = 'test.png'


def real_time(model):
    """
    Tests in real time a model.

    :param model: a keras model
        Only convolutional models are currently supported.
    """
    while True:

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

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        specgram_lbrs(audiopath=WAVE_OUTPUT_FILENAME, name='test', y_axis='log')
        p = np.round(model.predict(np.asarray([img_load(IMG_OUTPUT_FILENAME)])))
        if p[0][0] == 1:
            print('en')
        elif p[0][1] == 1:
            print('es')
        elif p[0][2] == 1:
            print('pt')


if __name__ == '__main__':
    # todo: add command line arguments
    p = Process(real_time(load_model('models/scratch_model/conv.h5')))
    p.start()
    input('Press any key to stop')
    p.terminate()
    os.remove(WAVE_OUTPUT_FILENAME)
    os.remove(IMG_OUTPUT_FILENAME)
