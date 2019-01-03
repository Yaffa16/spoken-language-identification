from matplotlib import pyplot as plt
from keras import optimizers
from keras.models import Sequential, Input
from keras.layers import (Conv2D,
                          MaxPooling2D,
                          Dense,
                          Flatten,
                          Dropout)
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from core.callbacks import EarlyStoppingRange
import keras
import imageio
import numpy as np
import tensorflow as tf
import talos
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self._config = config
        self._model = None

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    @property
    def model(self):
        return self._model

    @property
    def config(self):
        return self._config

    def train(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.train(), self

    @abstractmethod
    def build_model(self):
        raise NotImplementedError


class HyperParameterSearchable:

    def run(self):
        pass




