# Scratch for neural networks modeling

import matplotlib
# matplotlib.use("Agg")

from matplotlib import pyplot as plt
from util.batching.generator import Generator
from util.datasets.csv import CSVParser
from keras import optimizers
from keras.models import Sequential
from util.datasets.balance import balance_data
from keras.layers import (Conv2D,
                          MaxPooling2D,
                          Dense,
                          Flatten,
                          Dropout,
                          LSTM,
                          CuDNNLSTM)
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from core.callbacks import EarlyStoppingRange
from keras.callbacks import TensorBoard
from util.imgload import img_load
import numpy as np
import random
import time

# Image properties:
# width, height = 800, 513  # sox
width, height = 640, 480  # librosa
# width, height = 768, 256  # dft


def get_data(source: str, shuffle=True, **options):
    options.setdefault('train_proportion', 0.75)
    options.setdefault('val_proportion', 0.25)
    train_proportion = options['train_proportion']
    val_proportion = options['val_proportion']

    paths, labels = CSVParser(source)()
    paths, labels = balance_data(paths, labels)

    paths_labels = list(zip(paths, labels))
    if shuffle:
        random.shuffle(paths_labels)

    paths, labels = zip(*paths_labels)

    X_train = np.asarray(paths[0:int(train_proportion * len(paths))])
    y_train = np.asarray(labels[0:int(train_proportion * len(paths))])
    X_val = np.asarray(paths[int(train_proportion * len(paths)):
                              int((train_proportion +
                                   val_proportion) * len(paths))])
    y_val = np.asarray(labels[int(train_proportion * len(paths)):
                               int((train_proportion +
                                    val_proportion) * len(paths))])
    num_classes = np.unique(labels).shape[0]

    # Set up the labels
    le = preprocessing.LabelEncoder()
    le.fit(np.unique(labels))

    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    return num_classes, X_train, X_val, y_train, y_val


#                                   CONV MODEL
#   ----------------------------------------------------------------------------
def conv():
    # Model constants

    input_shape = (height, width, 1)
    optimizer = optimizers.SGD()

    epochs = 30

    # Set up the data
    num_classes, X_train, X_val, y_train, y_val = \
        get_data('data/dataset_test_1.csv')

    # Set up the generator
    training_gen = Generator(X_train, y_train, batch_size=16,
                             loader_fn=img_load)
    val_gen = Generator(X_val, y_val, batch_size=16, loader_fn=img_load)
    # Build model

    model = Sequential()

    # Convolutional layers:

    # Convolutional layer 1
    model.add(Conv2D(filters=16, kernel_size=(7, 7), strides=1,
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    # Convolutional layer 2
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

    # Convolutional layer 3
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

    # Convolutional layer 4
    model.add(Conv2D(128, (3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

    # Convolutional layer 5
    model.add(Conv2D(128, (3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

    # Convolutional layer 6
    model.add(Conv2D(256, (3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

    # Dense layers:

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # Classification layer:

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    plot_model(model, to_file='models/scratch_conv_model.png', show_shapes=True)

    print(model.summary())

    es = EarlyStoppingRange(monitor='acc', min_delta=0.01, patience=4,
                            verbose=2, mode='auto', min_val_monitor=0.8)

    tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16,
                     write_graph=True, write_grads=False, write_images=False,
                     embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None, embeddings_data=None)
    # Network training:
    H = model.fit_generator(generator=training_gen,
                            validation_data=val_gen,
                            epochs=epochs,
                            callbacks=[es, tb],
                            verbose=1)

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
    plt.savefig('001_scratch_conv_{}.png'.format(str(int(round(time.time() *
                                                               1000)))))

    print('[INFO] serializing network...')
    model.save('models/scratch_model/test001_{}.h5'.format(str(int(
        round(time.time() * 1000)))))
    print('DONE')


if __name__ == '__main__':
    # conv()
    from keras.models import load_model
    model = load_model('models/scratch_model/test001_1540234086146.h5')
    # Set up the data
    num_classes, X_train, X_val, y_train, y_val = \
        get_data('data/seconds_5_augment_data_rate16k.csv')
    #
    # # Set up the generator
    # gen = Generator(X_train, y_train, batch_size=16, loader_fn=img_load)
    # print(model.metrics_names)
    # print(model.evaluate_generator(generator=gen, verbose=1, workers=3,
    #                                use_multiprocessing=True))
    # for X, y in zip(X_train, y_train):
    #     p = np.round(model.predict(np.asarray([img_load(X)])))
    #     print(np.array_equal(p[0], y))
