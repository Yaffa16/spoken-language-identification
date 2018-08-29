"""
This module implements a pipeline utility for batching purposes.

# todo: doc test
"""

from abc import abstractmethod
import tensorflow as tf


class Pipeline:
    """A data pipeline using tensorflow.data.dataset"""

    def __init__(self, data, labels, source: str, batch_size: int,
                 loader_fn: callable=None, pre_process_fn: callable=None,
                 shuffle: bool=True, num_parallel_calls: int=None):
        """
        Initializes a new data pipeline.

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

        :param num_parallel_calls: int
            An int representing the number elements to process in parallel.
            If not specified, elements will be processed sequentially.
        """
        self.batch_size = batch_size
        self._source = source
        # Map source with file names
        self._paths = list(map(lambda x: source + '/' + x, data))
        self._labels = labels
        # Assign pre process function
        self._pre_process_fn = pre_process_fn
        # Assign loader function
        if loader_fn is not None:
            self.loader = loader_fn

        # Build dataset
        self._dataset = tf.data.Dataset.from_tensor_slices((self._paths,
                                                            self._labels))

        if shuffle:
            self._dataset = self._dataset.shuffle(len(data))

        # Map function to load data
        self._dataset = self._dataset.map(self.loader,
                                          num_parallel_calls=num_parallel_calls)
        # Map function to pre process data
        if pre_process_fn is not None:
            self._dataset = self._dataset.map(self._pre_process_fn,
                                              num_parallel_calls=
                                              num_parallel_calls)
        self._dataset = self._dataset.batch(batch_size)
        self._dataset = self._dataset.prefetch(1)

        # Create an initializable iterator from dataset
        iterator = self._dataset.make_initializable_iterator()
        x, y = iterator.get_next()
        iterator_init_op = iterator.initializer

        self._inputs = {'data': x, 'labels': y,
                        'iterator_init_op': iterator_init_op}

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

    def __call__(self, *args, **kwargs):
        """Short hand get the inputs."""
        return self.inputs

    @property
    def inputs(self):
        """Returns the iterable inputs"""
        return self._inputs
    
    @property
    def dataset(self):
        """Returns the dataset"""
        return self._dataset

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
