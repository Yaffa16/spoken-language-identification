# Scratch for neural networks modeling

from util.data.buffer import BufferThread
from matplotlib import pyplot as plt
import keras
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
from sklearn import preprocessing
from util.callbacks import CustomEarlyStopping
import imageio
import numpy as np
import warnings
import matplotlib
import os


matplotlib.use("Agg")

# Define some constants/variables:
hdf5_path = 'data/hdf5/topcoder/trainingdata/sox.hdf5'
data_path = 'data/spectrograms/topcoder/trainingdata/sox'
data_ids_labels_path = 'data/raw/topcoder/trainingData.csv'
train_ds_name = 'train_sox'
val_ds_name = 'val_sox'
test_ds_name = 'test_sox'
# Image properties:
# width, height = 800, 513  # sox
# width, height =   # matplotlib
width, height = 640, 480  # librosa
# width, height = 768, 256  # dft


# Define some functions/classes:
class ImagesBuffer(BufferThread):
    def loader(self, source_path: str, *args, **kwargs):
        i = imageio.imread(source_path)
        return np.asarray(i / 255)


def img_load(*a, **k):
    i = imageio.imread(*a, **k)
    i = np.asarray(i/255)
    if len(i.shape) > 2 and i.shape[2] > 1:
        i = i[:, :, 0]
    return i.reshape(i.shape[0], i.shape[1], 1)


# Network modeling:
def build(input_shape, classes, optimizer,
          dropout_rate=0, batch_normalization=False):

    model = Sequential()

    # Convolutional layers:

    # Convolutional layer 1
    model.add(Conv2D(filters=16, kernel_size=(7, 7), strides=1,
                     activation='relu',
                     input_shape=input_shape))
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

    model.add(Dense(classes, activation='softmax'))

    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Classification layer:

    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def main(X_train, y_train, X_val, y_val, X_test, y_test):

    print('Size of train data set: {}, size of test data set: {}'.
          format(X_train.shape[0], X_test.shape[0]))

    print('Shape of X train data set: {}, shape of y train data set: {}'.
          format(X_train.shape, y_train.shape))

    print('Shape of X test data set: {}, shape of y test data set: {}'.
          format(X_test.shape, y_test.shape))

    input_shape = height, width, 1

    # optimizer = optimizers.SGD(lr=0.3, decay=1e-6, momentum=0.9,
    # nesterov=True)
    optimizer = optimizers.Adadelta()
    model = build(input_shape=input_shape, classes=num_classes,
                  optimizer=optimizer, batch_normalization=False)

    print(model.summary())

    plot_model(model, to_file='models/scratch_model.png', show_shapes=True,
               show_layer_names=True)

    # Network training:

    # H = model.fit_generator # todo ?

    es = CustomEarlyStopping(monitor='val_acc', min_delta=0.01, patience=25,
                             verbose=2, mode='auto', min_val_monitor=0.8)
    es2 = keras.callbacks.EarlyStopping(monitor='val_los', patience=100,
                                        verbose=2, mode='auto', min_delta=0.01)

    H = model.fit(X_train, y_train, batch_size=16, epochs=epochs, verbose=1,
                  validation_data=(X_test, y_test), shuffle=False,
                  callbacks=[es])

    score = model.evaluate(X_val, y_val, verbose=0)
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
    plt.savefig('scratch_train.png')

    print('[INFO] serializing network...')
    model.save('models/scratch_model')
    print('DONE')


if __name__ == '__main__':

    # Model constants:
    batch_norm = True
    dropout = 0.5
    epochs = 1000

    # data_ids_labels = open(data_ids_labels_path, 'r').readlines()[1:]
    # labels = []
    # for line in data_ids_labels:
    #     labels.append(line.split(',')[1][:-2])
    # num_classes = np.unique(labels).shape[0]

    # serialize(os.listdir(data_path), labels)
    # main(load_data())

    data_ids_labels = open('data/spectrograms/topcoder_small/trainingData.csv',
                           'r').readlines()
    labels = []
    for line in data_ids_labels:
        labels.append(line.split(',')[1][:-2])

    num_classes = np.unique(labels).shape[0]

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))

    X = []
    for spec in os.listdir('data/spectrograms/topcoder_small/'
                           'trainingdata/librosa/mel'):
        X.append(img_load('data/spectrograms/topcoder_small/'
                          'trainingdata/librosa/mel/' + spec))

    assert len(X) == len(labels)

    X_train = np.asarray(X[0:int(0.8 * len(X))])
    y_train = np.asarray(labels[0:int(0.8 * len(labels))])
    X_val = np.asarray(X[int(0.8 * len(X)):int(0.9 * len(X))])
    y_val = np.asarray(labels[int(0.8 * len(X)):int(0.9 * len(X))])
    X_test = np.asarray(X[int(0.9 * len(X)):])
    y_test = np.asarray(labels[int(0.9 * len(labels)):])

    # Set up the labels
    y_test = le.transform(y_test)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # y_transform = to_categorical(le.transform(labels))
    # for i in range(len(labels)):
    #     if labels[i] == 'Adiokr':
    #         assert y_transform[i][0] == 1
    #     elif labels[i] == 'Aguaruna Awaj':
    #         assert y_transform[i][1] == 1
    #     else :
    #         assert y_transform[i][2] == 1

    main(X_train, y_train, X_val, y_val, X_test, y_test)
