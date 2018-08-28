"""This module implements basic operations to load images for model training"""

import imageio
import numpy as np


def img_load(*a, return_channels=True):
    i = imageio.imread(*a)
    i = np.asarray(i/255)
    if len(i.shape) > 2 and i.shape[2] > 1:
        i = i[:, :, 0]
    i = i.reshape(i.shape[0], i.shape[1], 1)
    if not return_channels:
        return i[:, :, 0]
    else:
        return i