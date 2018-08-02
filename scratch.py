# Scratch for neural networks modeling

from util.buffer import BufferThread
from matplotlib import pyplot as plt
from keras.utils.io_utils import HDF5Matrix
from random import shuffle
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import (Conv2D,
                          MaxPooling2D,
                          Dense,
                          Flatten,
                          Dropout)
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import imageio
import numpy as np
import queue
import h5py
import matplotlib
import os


matplotlib.use("Agg")

# Define some constants/variables:
BUFFER_SIZE = 1000
BATCH_SIZE = 1000
hdf5_path = 'data/hdf5/topcoder/trainingdata/sox.hdf5'
data_path = 'data/spectrograms/topcoder/trainingdata/sox'
data_ids_labels_path = 'data/raw/topcoder/trainingData.csv'
train_ds_name = 'train_sox'
val_ds_name = 'val_sox'
test_ds_name = 'test_sox'
# Image properties:
width, height = 513, 800


# Define some functions/classes:
class ImagesBuffer(BufferThread):
    def loader(self, source_path: str, *args, **kwargs):
        i = imageio.imread(source_path)
        return np.asarray(i / 255)


def img_load(*a, **k):
    i = imageio.imread(*a, **k)
    i = np.asarray(i/255)
    return i.reshape(i.shape[0], i.shape[1], 1)


def serialize(X_paths, y):
    data_and_labels = list(zip(X_paths, y))
    shuffle(data_and_labels)
    specs, y = zip(*data_and_labels)

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(y))

    # Divide the data into 60% train, 20% validation, and 20% test
    train_paths = specs[0:int(0.6 * len(specs))]
    y_train = y[0:int(0.6 * len(y))]
    val_paths = specs[int(0.6 * len(specs)):int(0.8 * len(specs))]
    y_val = y[int(0.6 * len(specs)):int(0.8 * len(specs))]
    test_paths = specs[int(0.8 * len(specs)):]
    y_test = y[int(0.8 * len(y)):]

    # Set up the labels
    y_test = le.transform(y_test)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Set the shape of the data
    train_shape = (len(train_paths), 513, 800, 1)
    val_shape = (len(val_paths), 513, 800, 1)
    test_shape = (len(test_paths), 513, 800, 1)

    f = h5py.File(hdf5_path, 'w')
    f.create_dataset(train_ds_name, shape=train_shape, dtype=np.int8)
    f.create_dataset(val_ds_name, shape=val_shape, dtype=np.int8)
    f.create_dataset(test_ds_name, shape=test_shape, dtype=np.int8)
    f.create_dataset('train_mean', shape=train_shape[1:], dtype=np.float32)
    f.create_dataset('train_labels', shape=(len(train_paths), num_classes),
                     dtype=np.int8)
    f['train_labels'][...] = y_train
    f.create_dataset("val_labels", shape=(len(val_paths), num_classes),
                     dtype=np.int8)
    f['val_labels'][...] = y_val
    f.create_dataset("test_labels", shape=(len(test_paths), num_classes),
                     dtype=np.int8)
    f['test_labels'][...] = y_test

    # A numpy array to save the mean of the images
    mean = np.zeros(train_shape[1:], np.float32)

    print('[INFO] serializing...')
    # Loop over train paths
    for i in range(len(train_paths)):
        if i % 1000 == 0 and i > 1:
            print('[INFO] train data: {}/{}'.format(i, len(train_paths)))

        img = img_load(data_path + '/' + train_paths[i])

        # <add any image pre-processing here>

        f['train_sox'][i, ...] = img
        mean += img / float(len(y_train))

    # Loop over val paths
    for i in range(len(val_paths)):
        if i % 1000 == 0 and i > 1:
            print('[INFO] val data: {}/{}'.format(i, len(val_paths)))

        img = img_load(data_path + '/' + val_paths[i])

        # <add any image pre-processing here>

        f['val_sox'][i, ...] = img

    # Loop over test paths
    for i in range(len(test_paths)):
        if i % 1000 == 0 and i > 1:
            print('[INFO] test data: {}/{}'.format(i, len(test_paths)))

        img = img_load(data_path + '/' + test_paths[i])

        # <add any image pre-processing here>

        f['test_sox'][i, ...] = img

    f["train_mean"][...] = mean
    f.close()


def load_data():
    X_train_ds = HDF5Matrix(hdf5_path, train_ds_name)
    y_train_ds = HDF5Matrix(hdf5_path, 'train_labels')
    X_val_ds = HDF5Matrix(hdf5_path, val_ds_name)
    y_val_ds = HDF5Matrix(hdf5_path, 'val_labels')
    X_test_ds = HDF5Matrix(hdf5_path, test_ds_name)
    y_test_ds = HDF5Matrix(hdf5_path, 'test_labels')

    return X_train_ds, y_train_ds, X_val_ds, y_val_ds, X_test_ds, y_test_ds


# Network modeling:
def build(input_shape, classes, optimizer,
          dropout_rate=0, batch_normalization=False):

    model = Sequential()

    # Convolutional layers:

    # Convolutional layer 1
    model.add(Conv2D(filters=16, kernel_size=(7, 7), strides=1,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    if batch_normalization:
        model.add(BatchNormalization())

    # Convolutional layer 2
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    if batch_normalization:
        model.add(BatchNormalization())

    # Convolutional layer 3
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    if batch_normalization:
        model.add(BatchNormalization())

    # Convolutional layer 4
    model.add(Conv2D(128, (3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    if batch_normalization:
        model.add(BatchNormalization())

    # Convolutional layer 5
    model.add(Conv2D(128, (3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    if batch_normalization:
        model.add(BatchNormalization())

    # Convolutional layer 6
    model.add(Conv2D(256, (3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
    if batch_normalization:
        model.add(BatchNormalization())

    # Dense layers:

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    if batch_normalization:
        model.add(BatchNormalization())

    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Classification layer:

    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print('Size of train data set: {}, size of test data set: {}'.
          format(X_train.shape[0], X_test.shape[0]))

    print('Shape of X train data set: {}, shape of y train data set: {}'.
          format(X_train.shape, y_train.shape))

    print('Shape of X test data set: {}, shape of y test data set: {}'.
          format(X_test.shape, y_test.shape))

    input_shape = width, height, 1

    sgd = optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    model = build(input_shape=input_shape, classes=num_classes, optimizer=sgd)

    print(model.summary())

    plot_model(model, to_file='models/scratch_model.png', show_shapes=True,
               show_layer_names=True)

    # Network training:

    # H = model.fit_generator # todo ?
    H = model.fit(X_train, y_train, batch_size=16, epochs=epochs, verbose=1,
                  validation_data=(X_test, y_test), shuffle=False)

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('[INFO] serializing network...')
    model.save('models/scratch_model')
    print('DONE')

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(fig_size=(10, 5))
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train.png')


if __name__ == '__main__':

    # Model constants:
    batch_norm = True
    dropout = 0.5
    epochs = 100

    data_ids_labels = open(data_ids_labels_path, 'r').readlines()[1:]
    labels = []
    for line in data_ids_labels:
        labels.append(line.split(',')[1][:-2])
    num_classes = np.unique(labels).shape[0]

    # serialize(os.listdir(data_path), labels)
    main()
