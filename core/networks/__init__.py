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


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None

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

    def build_model(self):
        raise NotImplementedError
