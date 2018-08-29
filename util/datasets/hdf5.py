from keras.utils.io_utils import HDF5Matrix
from random import shuffle
from keras.utils import to_categorical
from sklearn import preprocessing
import numpy as np
import h5py


class HDF5:
    """
    Represents data sets in hdf5 format.

    This class provides useful methods to save and retrieve data from hdf5
    files.
    """
    def __init__(self, path: str,
                 train_ds_name: str='train_dataset',
                 test_ds_name: str='test_dataset',
                 val_ds_name: str='val_dataset'):
        """
        :param path: str
            Path to the hdf5 file. The file may or may not exist in the path
            provided.
        :param train_ds_name: str
            Name of the train dataset.
        :param test_ds_name: str
            Name of the test dataset.
        :param val_ds_name: str
            Name of the validation dataset.
        """
        self.path = path
        self.train_ds_name = train_ds_name
        self.test_ds_name = test_ds_name
        self.val_ds_name = val_ds_name

    def serialize(self, x_paths, labels, source: str, loader: callable,
                  shape: tuple, split_train_test_val: tuple=(0.6, 0.2, 0.2),
                  shuffle_data: bool=True):
        """
        Saves data in the hdf5 file. The data is obtained on the source, and
        each data is read by the loader function provided.

        :param x_paths: list or ndarray
            List containing paths to the data.
        :param labels: list or ndarray
            List containing the respective labels of data.
        :param source: str
            Path to the source directory to get the data.
        :param loader: callable
            A callable to load data. For instance, an image loader reads an
            image and returns a numpy.ndarray to be read later.
        :param shape: tuple
            Shape of data.
        :param split_train_test_val: tuple
            Values that represents the split sizes to split the data into
            train, test and validation data sets.
        :param shuffle_data: bool
            Shuffles data before saving.
        """
        # Sets the size of each data set
        train_size = split_train_test_val[0]
        test_size = split_train_test_val[1]
        val_size = split_train_test_val[2]
        if train_size + test_size + val_size > 1.:
            raise ValueError('Invalid split sizes provided')

        # Zip paths and labels to shuffle
        data_and_labels = list(zip(x_paths, labels))
        if shuffle_data:
            shuffle(data_and_labels)
        data, labels = zip(*data_and_labels)

        # Call label encoder to process labels as one hot vectors
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(labels))
        num_classes = len(np.unique(labels))

        # Split data
        train_paths = data[0:int(train_size * len(data))]
        y_train = labels[0:int(train_size * len(labels))]
        val_paths = data[int(train_size * len(data)):int((train_size + val_size)
                                                         * len(data))]
        y_val = labels[int(train_size * len(data)):int(train_size + val_size
                                                       * len(data))]
        test_paths = data[int(train_size + val_size * len(data)):]
        y_test = labels[int(train_size + val_size * len(labels)):]

        # Set up the labels
        y_test = le.transform(y_test)
        y_train = le.transform(y_train)
        y_val = le.transform(y_val)
        y_test = to_categorical(y_test, num_classes=num_classes)
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)

        # Set the shape of the data
        train_shape = (len(train_paths), *shape)
        val_shape = (len(val_paths), *shape)
        test_shape = (len(test_paths), *shape)

        # Opens the hdf5 file in write mode
        f = h5py.File(self.path, 'w')
        # Creates the data set
        f.create_dataset(self.train_ds_name, shape=train_shape, dtype=np.int8)
        f.create_dataset(self.val_ds_name, shape=val_shape, dtype=np.int8)
        f.create_dataset(self.test_ds_name, shape=test_shape, dtype=np.int8)
        f.create_dataset('train_labels', shape=(len(train_paths), num_classes),
                         dtype=np.int8)
        f['train_labels'][...] = y_train
        f.create_dataset("val_labels", shape=(len(val_paths), num_classes),
                         dtype=np.int8)
        f['val_labels'][...] = y_val
        f.create_dataset("test_labels", shape=(len(test_paths), num_classes),
                         dtype=np.int8)
        f['test_labels'][...] = y_test

        # Saves the data
        print('[INFO] serializing...')
        # Loop over train paths
        for i in range(len(train_paths)):
            if i % 1000 == 0 and i > 1:
                print('[INFO] train data: {}/{}'.format(i, len(train_paths)))
            d = loader(source + '/' + train_paths[i])
            f[self.train_ds_name][i, ...] = d

        # Loop over val paths
        for i in range(len(val_paths)):
            if i % 1000 == 0 and i > 1:
                print('[INFO] val data: {}/{}'.format(i, len(val_paths)))
            d = loader(source + '/' + val_paths[i])
            f[self.val_ds_name][i, ...] = d

        # Loop over test paths
        for i in range(len(test_paths)):
            if i % 1000 == 0 and i > 1:
                print('[INFO] test data: {}/{}'.format(i, len(test_paths)))
            d = loader(source + '/' + test_paths[i])
            f[self.test_ds_name][i, ...] = d
        f.close()

    def load_data(self) -> tuple:
        """
        Loads the saved data.

        :return: tuple
            A tuple with the following read datasets:
            - Train dataset
            - Train labels
            - Validation dataset
            - Validation labels
            - Test dataset
            - Test labels
            The type of these objects are HDF5Matrix.

            .: see keras.utils.io_utils.HDF5Matrix
        """
        X_train_ds = HDF5Matrix(self.path, self.train_ds_name)
        y_train_ds = HDF5Matrix(self.path, 'train_labels')
        X_val_ds = HDF5Matrix(self.path, self.val_ds_name)
        y_val_ds = HDF5Matrix(self.path, 'val_labels')
        X_test_ds = HDF5Matrix(self.path, self.test_ds_name)
        y_test_ds = HDF5Matrix(self.path, 'test_labels')

        return X_train_ds, y_train_ds, X_val_ds, y_val_ds, X_test_ds, y_test_ds
