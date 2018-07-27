"""
This module implements a buffer utility for batching purposes. With large
amounts of data, it is possible to queue the data in a separated thread.
TODO: shuffle
"""

from abc import abstractmethod
import threading
import numpy
import queue
import os


class BufferThread(threading.Thread):
    """A thread which buffers data in a queue"""

    def __init__(self, source: str, buffer: queue, data_ids_labels: list=None,
                 group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        """
        Initializes a new buffer thread.

        :param data_ids_labels: list
            List containing the tuples with the identification and label of each
            instance of data.

            Expected format of file:
            'name_of_file_1.ext', ..., ...
            'name_of_file_2.ext', ..., ...
            ...
            'name_of_file_n.ext', ..., ...
            *ext is the file extension.

            Default to none, all files in the source will be processed.

        :param source: str
            Path to the raw data.
        :param buffer:
            A queue to buffer data.

            Each read data will be put in the buffer as tuples. Each tuple have
            the following format:
            ('name/id_of_file', 'read_object')

        See threading.Thread for more information.
        """
        super(BufferThread, self).__init__()

        self.target = target
        self.name = name
        self._lock = threading.Lock()
        self._source = source
        self._current = 0
        self._ids_files = []
        self._buffer = buffer
        self._exit_request = False

        if data_ids_labels is not None:
            instances_ids = []
            for line in data_ids_labels:
                data = line.split(',')
                instances_ids.append(data[0][:-4])

            for file in os.listdir(source):
                if file[:-4] in instances_ids:
                    # Append file name, file name + extension:
                    self._ids_files.append((file[:-4], file))
                    instances_ids.remove(file[:-4])  # + speed
        else:
            for file in os.listdir(source):
                self._ids_files.append((file[:-4], file))

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
        raise NotImplementedError('Loader not implemented. Must implement a '
                                  'loader for correct operation of a buffer.')

    def run(self):
        """
        The run method will execute after the buffer's thread initializes.

        It will be executed while exists data to be read.
        """
        while self._current < len(self._ids_files) and not self._exit_request:
            with self._lock:
                if not self._buffer.full():
                    self._buffer.put(
                        (self._ids_files[self._current][0],
                         self.loader(self._source + '/' +
                                     self._ids_files[self._current][1])))
                    self._current += 1

    def __call__(self, *args, **kwargs):
        """Short hand to start the thread."""
        self.start()

    def get_batch(self, size: int) -> (numpy.ndarray, numpy.ndarray):
        """
        Get a batch of data.

        :param size: int
            Size of the batch
        :return: (numpy.ndarray, numpy.ndarray)
            Tuple of arrays. The first array represents the names/ids of data,
            and the second array represents the data itself.

        Note: in case the size of the batch is greater than the amount of data
        available, this method will continuously attempt to fill the necessary
        amount of data, or, if the thread is not alive, only the read amount
        will be returned.
        """
        ids = []
        data = []
        while len(data) < size:
            if self._buffer.empty() and not self.is_alive():
                break
            i, d = self._buffer.get()
            ids.append(i)
            data.append(d)
        return numpy.asarray(ids), numpy.asarray(data)

    def stop(self):
        """
        Stops the buffer.

        Note: this will not stop the thread abruptly."""
        self._exit_request = True
