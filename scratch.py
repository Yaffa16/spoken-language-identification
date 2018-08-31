# Scratch for neural networks modeling

import matplotlib
# matplotlib.use("Agg")

from matplotlib import pyplot as plt
from util.batching.generator import Generator
from util.datasets.csv import CSVParser
from keras import optimizers
from keras.models import Sequential, Input, load_model
from keras_tqdm import TQDMCallback
from tqdm import tqdm
from util.datasets.balance import balance_data
from keras.layers import (Conv2D,
                          MaxPooling2D,
                          Dense,
                          Flatten,
                          Dropout,
                          LSTM,
                          CuDNNLSTM)
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from core.callbacks import EarlyStoppingRange
from util.imgload import img_load
from util.datasets.numpyloader import load_dataset
from qrnn import QRNN
import keras
import numpy as np
import tensorflow as tf
import random
import librosa
import glob
import sys

# todo: implement model in a module
# todo: parse json files to get the hyper parameters
# todo: train model with all data

# Image properties:
# width, height = 800, 513  # sox
width, height = 640, 480  # librosa
# width, height = 768, 256  # dft


# Define some functions/classes:
# class ImagesBuffer(BufferThread):
#     def loader(self, source_path: str, *args, **kwargs):
#         i = imageio.imread(source_path)
#         return np.asarray(i / 255)

#
# class printbatch(keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs={}):
#         print(logs)
#     def on_epoch_end(self, epoch, logs={}):
#         print(logs)


def parse_function(filename, label):
    # print(filename, label)
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # image = tf.image.resize_images(image, [64, 64])
    # return image, label

#                                   CONV MODEL
#   ----------------------------------------------------------------------------


def conv():

    # Model constants

    input_shape = (height, width, 1)

    optimizer = optimizers.Adadelta()

    batch_normalization = True

    dropout_rate = 0.4

    epochs = 30

    # Set up the data

    # source = 'data/dataset_norm_lbrsdftlog.csv'
    source = 'data/dataset_topcoder_lbrsdftlog.csv'
    paths, labels = CSVParser(source)()
    paths, labels = balance_data(paths, labels)  # todo: check others

    paths_labels = list(zip(paths, labels))
    random.shuffle(paths_labels)

    paths, labels = zip(*paths_labels)

    X_train = np.asarray(paths[0:int(0.5 * len(paths))])
    y_train = np.asarray(labels[0:int(0.5 * len(paths))])
    X_test = np.asarray(paths[int(0.5 * len(paths)):int(0.75 * len(paths))])
    y_test = np.asarray(labels[int(0.5 * len(paths)):int(0.75 * len(paths))])
    X_val = np.asarray(paths[int(0.75 * len(paths)):])
    y_val = np.asarray(labels[int(0.75 * len(paths)):])
    num_classes = np.unique(labels).shape[0]

    # Set up the labels
    le = preprocessing.LabelEncoder()

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Set up the generator
    training_gen = Generator(X_train, y_train, batch_size=16,
                             loader_fn=img_load)
    validation_gen = Generator(X_val, y_val, batch_size=16,
                               loader_fn=img_load)
    test_gen = Generator(X_test, y_test, batch_size=16,
                         loader_fn=img_load)

    # Build model

    model = Sequential()

    # Convolutional layers:

    # Convolutional layer 1
    model.add(Conv2D(filters=16, kernel_size=(7, 7), strides=1,
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
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
    model.add(Dense(512, activation='relu'))

    if batch_normalization:
        model.add(BatchNormalization())

    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Classification layer:

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    plot_model(model, to_file='models/scratch_conv_model.png', show_shapes=True)

    print(model.summary())

    # Network training:

    es = EarlyStoppingRange(monitor='acc', min_delta=0.01, patience=4,
                            verbose=2, mode='auto', min_val_monitor=0.8)
    # es = keras.callbacks.EarlyStopping(monitor='loss', patience=2,
    #                                    verbose=2, mode='auto')
    mchkpt = keras.callbacks.ModelCheckpoint('models/scratch_model/'
                                             'checkpoints/model_checkpoint.h5')

    H = model.fit_generator(generator=training_gen,
                            validation_data=validation_gen,
                            epochs=epochs,
                            callbacks=[es],
                            verbose=1)

    score = model.evaluate_generator(generator=test_gen, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    N = len(H.epoch)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train_conv_tpc.png')

    print('[INFO] serializing network...')
    model.save('models/scratch_model/conv.h5')
    print('DONE')

#                                   LSTM MODEL
#   ----------------------------------------------------------------------------


def lstm():
    # Model constants

    input_shape = (height, width)  # todo: check input_shape, steps

    optimizer = optimizers.Adadelta()

    batch_normalization = True

    dropout_rate = 0.4

    epochs = 20

    # Set up the data

    source = 'data/dataset_norm_lbrsdftlog.csv'
    paths, labels = CSVParser(source)()
    paths, labels = balance_data(paths, labels)
    paths_labels = list(zip(paths, labels))
    random.shuffle(paths_labels)
    paths, labels = zip(*paths_labels)
    X_train = np.asarray(paths[0:int(0.5 * len(paths))])
    y_train = np.asarray(labels[0:int(0.5 * len(paths))])
    X_test = np.asarray(paths[int(0.5 * len(paths)):int(0.75 * len(paths))])
    y_test = np.asarray(labels[int(0.5 * len(paths)):int(0.75 * len(paths))])
    X_val = np.asarray(paths[int(0.75 * len(paths)):])
    y_val = np.asarray(labels[int(0.75 * len(paths)):])
    num_classes = np.unique(labels).shape[0]

    # Set up the labels
    le = preprocessing.LabelEncoder()

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Set up the generator
    training_gen = Generator(X_train, y_train, batch_size=16,
                             loader_fn=img_load, return_channels=False)
    validation_gen = Generator(X_val, y_val, batch_size=16,
                               loader_fn=img_load, return_channels=False)
    test_gen = Generator(X_test, y_test, batch_size=16,
                         loader_fn=img_load, return_channels=False)

    # Build model

    model = Sequential()
    model.add(CuDNNLSTM(400, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(200, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(100, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    plot_model(model, to_file='models/scratch_lstm_model.png', show_shapes=True)

    es = EarlyStoppingRange(monitor='acc', min_delta=0.01, patience=4,
                            verbose=2, mode='auto', min_val_monitor=0.8)

    H = model.fit_generator(generator=training_gen,
                            validation_data=validation_gen,
                            epochs=epochs,
                            callbacks=[es],
                            verbose=1)

    score = model.evaluate_generator(generator=test_gen, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    N = len(H.epoch)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train_lstm_tpc.png')

    print('[INFO] serializing network...')
    model.save('models/scratch_model/lstm.h5')
    print('DONE')

#                           LSTM MFCC MODEL (FIT GENERATOR)
#   ----------------------------------------------------------------------------


def lstm_mfcc_fgen():
    # Model constants

    input_shape = 20, 173

    optimizer = optimizers.Adadelta()

    batch_normalization = True

    dropout_rate = 0.4

    epochs = 60

    # Set up the data

    source = 'data/dataset_mfcc_topcoder.csv'
    paths, labels = CSVParser(source)()
    paths, labels = balance_data(paths, labels)
    paths_labels = list(zip(paths, labels))
    random.shuffle(paths_labels)
    paths, labels = zip(*paths_labels)
    X_train = np.asarray(paths[0:int(0.5 * len(paths))])
    y_train = np.asarray(labels[0:int(0.5 * len(paths))])
    X_test = np.asarray(paths[int(0.5 * len(paths)):int(0.75 * len(paths))])
    y_test = np.asarray(labels[int(0.5 * len(paths)):int(0.75 * len(paths))])
    X_val = np.asarray(paths[int(0.75 * len(paths)):])
    y_val = np.asarray(labels[int(0.75 * len(paths)):])
    num_classes = np.unique(labels).shape[0]

    # Set up the labels
    le = preprocessing.LabelEncoder()

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Set up the generator
    training_gen = Generator(X_train, y_train, batch_size=128,
                             loader_fn=np.load,
                             expected_shape=input_shape,
                             not_found_ok=True)
    validation_gen = Generator(X_val, y_val, batch_size=128,
                               loader_fn=np.load,
                               expected_shape=input_shape,
                               not_found_ok=True)
    test_gen = Generator(X_test, y_test, batch_size=128,
                         loader_fn=np.load,
                         expected_shape=input_shape,
                         not_found_ok=True)

    # Build model

    model = Sequential()

    model.add(CuDNNLSTM(400, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(200, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(100, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    plot_model(model, to_file='models/scratch_lstm_mfcc_model.png',
               show_shapes=True)

    # es = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)
    es = EarlyStoppingRange(monitor='acc', min_delta=0.01, patience=2,
                            verbose=2, mode='auto', min_val_monitor=0.8)

    H = model.fit_generator(generator=training_gen,
                            validation_data=validation_gen,
                            epochs=epochs,
                            callbacks=[es],
                            verbose=1)
    score = model.evaluate_generator(generator=test_gen, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    N = len(H.epoch)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train_lstm_mfcc_tpc.png')

    print('[INFO] serializing network...')
    model.save('models/scratch_model/lstm_mfcc.h5')
    print('DONE')

#                               QRNN MFCC MODEL
#   ----------------------------------------------------------------------------


def qrnn_mfcc():
    # Model constants

    input_shape = 20, 173

    optimizer = optimizers.Adadelta()

    batch_normalization = True

    dropout_rate = 0.4

    epochs = 60

    # Set up the data

    print('> Loading data')
    # data, labels = load_dataset('data/dataset_mfcc_norm.csv',
    #                             expected_shape=input_shape,
    #                             verbose=True)
    data, labels = load_dataset('data/dataset_mfcc_topcoder')
    print('Done')

    data_and_labels = list(zip(data, labels))
    random.shuffle(data_and_labels)
    data, labels = zip(*data_and_labels)

    X_train = np.asarray(data[0:int(0.7 * len(data))])
    y_train = np.asarray(labels[0:int(0.7 * len(labels))])
    X_test = np.asarray(data[int(0.7 * len(data)):int(0.85 * len(data))])
    y_test = np.asarray(labels[int(0.7 * len(labels)):int(0.85 * len(labels))])
    X_val = np.asarray(data[int(0.85 * len(data)):])
    y_val = np.asarray(labels[int(0.85 * len(labels)):])

    num_classes = np.unique(labels).shape[0]
    # Set up the labels
    le = preprocessing.LabelEncoder()

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Build model

    model = Sequential()
    model.add(QRNN(400, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(QRNN(200, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(QRNN(100, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    plot_model(model, to_file='models/scratch_qrnn_mfcc_model.png',
               show_shapes=True)

    es = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

    H = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val),
                  epochs=epochs, callbacks=[es], verbose=1, batch_size=128)

    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    N = len(H.epoch)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train_qrnn_mfcc_tpc.png')

    print('[INFO] serializing network...')
    model.save('models/scratch_model/qrnn_mfcc.h5')
    print('DONE')

#                                   QRNN MODEL
#   ----------------------------------------------------------------------------

def qrnn():
    # Model constants

    input_shape = (height, width)

    optimizer = optimizers.Adadelta()

    batch_normalization = True

    dropout_rate = 0.4

    epochs = 20

    # Set up the data

    source = 'data/dataset_topcoder_lbrsdftlog.csv'
    paths, labels = CSVParser(source)()
    paths_labels = list(zip(paths, labels))
    random.shuffle(paths_labels)
    paths, labels = zip(*paths_labels)
    X_train = np.asarray(paths[0:int(0.5 * len(paths))])
    y_train = np.asarray(labels[0:int(0.5 * len(paths))])
    X_test = np.asarray(paths[int(0.5 * len(paths)):int(0.75 * len(paths))])
    y_test = np.asarray(labels[int(0.5 * len(paths)):int(0.75 * len(paths))])
    X_val = np.asarray(paths[int(0.75 * len(paths)):])
    y_val = np.asarray(labels[int(0.75 * len(paths)):])
    num_classes = np.unique(labels).shape[0]

    # Set up the labels
    le = preprocessing.LabelEncoder()

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Set up the generator
    training_gen = Generator(X_train, y_train, batch_size=16,
                             loader_fn=img_load, return_channels=False)
    validation_gen = Generator(X_val, y_val, batch_size=16,
                               loader_fn=img_load, return_channels=False)
    test_gen = Generator(X_test, y_test, batch_size=16,
                         loader_fn=img_load, return_channels=False)

    # Build model

    model = Sequential()
    model.add(QRNN(400, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(QRNN(200, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(QRNN(100, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    plot_model(model, to_file='models/scratch_qrnn_model.png', show_shapes=True)

    es = EarlyStoppingRange(monitor='acc', min_delta=0.01, patience=25,
                            verbose=2, mode='auto', min_val_monitor=0.8)

    H = model.fit_generator(generator=training_gen,
                            validation_data=validation_gen,
                            epochs=epochs,
                            callbacks=[es],
                            verbose=1)

    score = model.evaluate_generator(generator=test_gen, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    N = len(H.epoch)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train_qrnn_tpc.png')

    print('[INFO] serializing network...')
    model.save('models/scratch_model/qrnn.h5')
    print('DONE')

#                                   LSTM MFCC MODEL
#   ----------------------------------------------------------------------------

def lstm_mfcc():
    # Model constants

    input_shape = 20, 173

    optimizer = optimizers.Adadelta()

    batch_normalization = True

    dropout_rate = 0.4

    epochs = 60

    # Set up the data

    print('> Loading data')
    # data, labels = load_dataset('data/dataset_mfcc_norm.csv',
    #                             expected_shape=input_shape)

    data, labels = load_dataset('data/dataset_mfcc_topcoder.csv',
                                expected_shape=input_shape,
                                not_found_ok=True,
                                verbose=True)
    print(data, labels)
    print('Done')

    data_and_labels = list(zip(data, labels))
    random.shuffle(data_and_labels)

    data, labels = zip(*data_and_labels)

    X_train = np.asarray(data[0:int(0.7 * len(data))])
    y_train = np.asarray(labels[0:int(0.7 * len(labels))])
    X_test = np.asarray(data[int(0.7 * len(data)):int(0.85 * len(data))])
    y_test = np.asarray(labels[int(0.7 * len(labels)):int(0.85 * len(labels))])
    X_val = np.asarray(data[int(0.85 * len(data)):])
    y_val = np.asarray(labels[int(0.85 * len(labels)):])

    num_classes = np.unique(labels).shape[0]
    # Set up the labels
    le = preprocessing.LabelEncoder()

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Build model

    model = Sequential()
    model.add(CuDNNLSTM(400, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(200, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(CuDNNLSTM(100, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    plot_model(model, to_file='models/scratch_lstm_mfcc_model.png',
               show_shapes=True)

    es = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

    H = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val),
                  epochs=epochs, callbacks=[es], verbose=1, batch_size=128)

    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    N = len(H.epoch)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train_lstm_mfcc_tpc.png')

    print('[INFO] serializing network...')
    model.save('models/scratch_model/lstm_mfcc.h5')
    print('DONE')


if __name__ == '__main__':
    functions = [lstm_mfcc_fgen, conv, lstm, qrnn, lstm_mfcc, qrnn_mfcc]
    functions[int(sys.argv[1])]()
