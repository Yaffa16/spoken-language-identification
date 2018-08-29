"""
This module implements a data generator utility based on the
keras.utils.Sequence

A good usage of this module is in the fit_generator method of Keras library.
"""

from abc import abstractmethod
import keras
import random
import numpy


class Generator(keras.utils.Sequence):
    """A generator which provides batches of data"""

    def __init__(self, paths, labels, batch_size: int,
                 loader_fn: callable = None, pre_process_fn: callable = None,
                 shuffle: bool = True, balance_samples=False, **loader_kw):
        """
        Initializes a generator.

        Is is necessary to provide or implement a loader function which loads
        data. The pre processing function is optional, and will be executed
        always before returning a new batch of data.

        :param paths:
            List containing the data file paths to load.

        :param labels:
            List containing the labels of each data.

        :param batch_size: int
            The size of the batch to queue.

        :param loader_fn: callable(source_path: str)
            A function to load the data, if None, must override the loader
            function.

        :param pre_process_fn: callable(data) -> new_data
            A function to pre process the data after it is loaded, and before
            returning the batch. Optional.

        :param shuffle: bool
            If true, shuffle the paths before loading them.

        :param balance_samples: bool
            If True, balance data to be representative. Default to False, no
            balancing will be applied, and the data will be considered as it is.

            Note: When balancing the data, the number os data and labels tends
            to decrease because the extra instances of some classes will be
            removed.

        :param loader_kw: Additional kwargs to be passed on to the loader
            function.
        """

        self._paths = paths
        self._labels = labels
        self._pre_process_fn = pre_process_fn
        self._batch_size = batch_size
        self._loaderkw = loader_kw
        if loader_fn is not None:
            self.loader = loader_fn

        if balance_samples:
            dataset = list(zip(self._paths, self._labels))
            unique, counts = numpy.unique(self._labels, return_counts=True)
            max_n_instances = min(counts)
            paths_per_label = dict()
            for label in unique:
                paths_per_label[label] = list(map(lambda d: d[1] == label,
                                                  dataset))
            self._paths = []
            for label in paths_per_label:
                self._paths.append(paths_per_label[label][:max_n_instances])

        if shuffle:
            dataset = list(zip(self._paths, self._labels))
            random.shuffle(dataset)
            self._paths, self._labels = zip(*dataset)

    @abstractmethod
    def loader(self, source_path: str, *args, **kwargs):
        """
        A loader of data. Must implement a loader for correct operation of the
        buffer.

        A loader accepts the source path of the file and returns the read data.

        :param source_path: str
            The path of the source to load.
        :param args: (optional)
            Additional args can be passed on to the loader function.
        :param kwargs: (optional)
            Additional kwargs can be passed on to internal function behavior.
        :return:
            The read object.
        """
        raise NotImplementedError('Loader not implemented. Must implement '
                                  'a loader for correct operation.')

    def __getitem__(self, index) -> (numpy.ndarray, numpy.ndarray):
        """
        Gets a batch of data.

        :return: (numpy.ndarray, numpy.ndarray)
            Tuple of arrays. The first array represents the data, and the
            second represents the labels.

        .. note: in case the size of the batch is greater than the amount of
        data available, only the read amount will be returned.
        """
        # Generate indexes of the batch
        paths = self._paths[(index*self._batch_size):
                            ((index+1)*self._batch_size)]
        labels = self._labels[(index*self._batch_size):
                              ((index+1)*self._batch_size)]
        # Fill batches
        data = list(map(lambda p: self.loader(p, **self._loaderkw), paths))
        if self._pre_process_fn is not None:
            data = self._pre_process_fn(data)
        return numpy.asarray(data), numpy.asarray(labels)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(numpy.floor(len(self.paths) / self._batch_size))

    @property
    def size(self) -> int:
        """Returns the total quantity of original data"""
        return len(self._paths)

    @property
    def paths(self) -> list:
        """Returns a list containing the paths of data files"""
        return self._paths

    @property
    def labels(self):
        """Returns a list containing the labels of each instance of data"""
        return self._labels

    def on_epoch_end(self):
        pass
