"""
This module implements a buffer utility for batching purposes. With large
amounts of data, it is possible to queue the data in a separated thread.

# todo: doc test
>>> for e in range(nb_epoch):
>>>    print("epoch %d" % e)
>>>    for X_train, Y_train in ImageNet():
>>>        for X_batch, Y_batch in datagen.flow(X_train, Y_train,
>>>                                             batch_size=32):
>>>            loss = model.train(X_batch, Y_batch)
"""

from abc import abstractmethod
import random
import threading
import numpy
import queue
import os


class BufferThread(threading.Thread):
    """A thread which buffers data in a queue"""

    def __init__(self, data, labels, source: str, batch_size: int,
                 loader_fn: callable=None, pre_process_fn: callable=None,
                 shuffle: bool=True):
        """
        Initializes a new buffer thread.

        :param source: str
            Path to the raw data directory.

        :param data:
            List containing the data file names to load.

        :param labels:
            List containing the labels of each data.

        :param batch_size: int
            The size of the batch to queue.

        :param loader_fn: callable(source_path: str)
            A function to load the data, if None, must override the loader
            function.

        :param pre_process_fn: callable(data, labels) -> new_data, new_labels
            A function to pre process the data after it is loaded, and before
            returning the batch. Optional.

        :param shuffle: bool
            If true, shuffle the data before load it.

        See threading.Thread for more information.
        """
        super(BufferThread, self).__init__()

        self._paths = list(map(lambda x: source + '/' + x, data))
        self._labels = labels
        self._lock = threading.Lock()
        self._source = source
        self._current = 0
        self._buffer = queue.Queue(batch_size)
        self._exit_request = False
        self._pre_process_fn = pre_process_fn
        if loader_fn is not None:
            self.loader = loader_fn

        if shuffle:  # todo: check if this part of code is working
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

    def run(self):
        """
        The run method will execute after the buffer's thread starts.

        It will be executed while exists data to be read.
        """
        while self._current < len(self._paths) and not self._exit_request:
            with self._lock:
                if not self._buffer.full():
                    self._buffer.put(
                        (self.loader(self._paths[self._current]),
                         self._labels[self._current]))
                    self._current += 1

    def __call__(self, *args, **kwargs):
        """Short hand to start the thread."""
        self.start()

    def get_batch(self, size: int) -> (numpy.ndarray, numpy.ndarray):
        """
        Get a batch of data.

        If the pre processing function is implemented, the returned batch will
        be passed through the implemented method before returning.

        :param size: int
            Size of the batch
        :return: (numpy.ndarray, numpy.ndarray)
            Tuple of arrays. The first array represents the data, and the
            second represents the labels.

        :Example:
        # >>>

        .. note: in case the size of the batch is greater than the amount of
        data available, this method will continuously attempt to fill the
        necessary amount of data, or, if the thread is not alive, only the read
        amount will be returned.
        """
        data = []
        labels = []
        while len(data) < size:
            if self._buffer.empty() and not self.is_alive():
                break
            d, label = self._buffer.get()
            data.append(d)
            labels.append(label)
        if self._pre_process_fn is not None:
            data, labels = self._pre_process_fn(data, labels)
        return numpy.asarray(data), numpy.asarray(labels)

    def stop(self):
        """
        Stops the buffer.

        Note: this will not stop the thread abruptly."""
        self._exit_request = True

    @property
    def remaining(self) -> int:
        """Returns the amount of files remaining to be buffered."""
        return len(self._paths) - self._current

    @property
    def source(self) -> str:
        """Returns the source path of data"""
        return self._source

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

    def __len__(self) -> int:
        """Returns buffers's size"""
        return self._buffer.qsize()
