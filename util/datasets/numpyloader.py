"""
This module provides a simple way to get the dataset saved in npy format.
"""
from util.datasets.csv import CSVParser as Parser
import numpy as np


def load_dataset(csv_dataset_path: str, expected_shape: tuple=None):
    """
    Loads a dataset saved in numpy binary files (.npy format).

    :param csv_dataset_path: str
        Path to the CSV file containing paths and labels.
        Expected format:
        path_1.npy, label_1
        path_2.npy, label_2
        ...
        path_n.npy, label_n

    :param expected_shape: tuple
        Check if the shape of each loaded data is in the provided format. If
        not, the data will be ignored.

    :return: tuple (numpy.ndarray, numpy.ndarray)
        A tuple with the data loaded and the respective labels.
    """
    paths, labels = Parser(csv_dataset_path)()
    X = []
    y = []
    for path, label in zip(paths, labels):
        data = np.load(path)
        if expected_shape is not None and data.shape != expected_shape:
            continue
        X.append(data)
        y.append(label)

    return np.asarray(X), np.asarray(y)
