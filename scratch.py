# Scratch for neural networks modeling

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from util.batching.generator import Generator
from util.data.csv import CSVParser
from keras import optimizers
from keras.models import Sequential, Input, load_model
from keras_tqdm import TQDMCallback
from keras.layers import (Conv2D,
                          MaxPooling2D,
                          Dense,
                          Flatten,
                          Dropout,
                          LSTM)
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from core.callbacks import EarlyStoppingRange
from util.imgload import img_load
import keras
import numpy as np
import tensorflow as tf
import random
import librosa

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
    return image, label


def conv():

    # Model constants

    input_shape = (height, width, 1)

    optimizer = optimizers.Adadelta()

    batch_normalization = True

    dropout_rate = 0.4

    epochs = 5

    # Set up the data

    source = 'data/dataset_norm_lbrsdftlog.csv'
    paths, labels = CSVParser(source)()
    paths_labels = list(zip(paths, labels))
    random.shuffle(paths_labels)
    paths, labels = zip(*paths_labels)
    X_train = np.asarray(paths[0:int(0.5 * len(paths))])
    y_train = np.asarray(labels[0:int(0.5 * len(labels))])
    X_test = np.asarray(paths[int(0.5 * 0.75):])
    y_test = np.asarray(labels[int(0.5 * 0.75):])
    X_val = np.asarray(paths[int(0.75 * len(paths)):])
    y_val = np.asarray(labels[int(0.75 * len(labels)):])
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
                             loader_fn=img_load, balance_samples=True)
    validation_gen = Generator(X_val, y_val, batch_size=16,
                               loader_fn=img_load, balance_samples=True)
    test_gen = Generator(X_test, y_test, batch_size=16,
                         loader_fn=img_load, balance_samples=True)

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

    plot_model(model, to_file='models/scratch_model.png', show_shapes=True,
               show_layer_names=True)

    print(model.summary())

    # Network training:

    # es = EarlyStoppingRange(monitor='acc', min_delta=0.01, patience=25,
    #                         verbose=2, mode='auto', min_val_monitor=0.8)
    es = keras.callbacks.EarlyStopping(monitor='loss', patience=2,
                                       verbose=2, mode='auto')
    mchkpt = keras.callbacks.ModelCheckpoint('models/scratch_model/'
                                             'model_checkpoint.h5')

    H = model.fit_generator(generator=training_gen,
                            validation_data=validation_gen,
                            epochs=epochs,
                            callbacks=[es, mchkpt],
                            verbose=1)

    score = model.evaluate_generator(generator=test_gen, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 5))
    N = len(H.epoch)
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train.png')

    print('[INFO] serializing network...')
    model.save('models/scratch_model/conv.h5')
    print('DONE')


def lstm():
    # Model constants

    input_shape = (height, width)

    optimizer = optimizers.Adadelta()

    batch_normalization = True

    dropout_rate = 0.4

    epochs = 5

    # Set up the data

    source = 'data/dataset_norm_lbrsdftlog.csv'
    paths, labels = CSVParser(source)()
    paths_labels = list(zip(paths, labels))
    random.shuffle(paths_labels)
    paths, labels = zip(*paths_labels)
    X_train = np.asarray(paths[0:int(0.5 * len(paths))])
    y_train = np.asarray(labels[0:int(0.5 * len(labels))])
    X_test = np.asarray(paths[int(0.5 * 0.75):])
    y_test = np.asarray(labels[int(0.5 * 0.75):])
    X_val = np.asarray(paths[int(0.75 * len(paths)):])
    y_val = np.asarray(labels[int(0.75 * len(labels)):])
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
                             loader_fn=img_load, return_channels=False,
                             balance_samples=False)
    validation_gen = Generator(X_val, y_val, batch_size=16,
                               loader_fn=img_load, return_channels=False,
                               balance_samples=False)
    test_gen = Generator(X_test, y_test, batch_size=16,
                         loader_fn=img_load, return_channels=False,
                         balance_samples=False)

    # Build model

    model = Sequential()
    model.add(LSTM(400, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LSTM(200, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LSTM(100, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    es = keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1)

    H = model.fit_generator(generator=training_gen,
                            validation_data=validation_gen,
                            epochs=epochs,
                            callbacks=[es],
                            verbose=1)

    score = model.evaluate_generator(generator=test_gen, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('[INFO] serializing network...')
    model.save('models/scratch_model/lstm.h5')
    print('DONE')

lstm_nfcc():
    librosa.feature.mfcc


if __name__ == '__main__':
    # conv()
    # lstm()
    # lstm_nfcc()