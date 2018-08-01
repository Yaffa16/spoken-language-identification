# First scratch_model for neural networks modeling

from util.buffer import BufferThread
import imageio
import numpy as np
import queue

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


# 1. Define some constants/variables:
BUFFER_SIZE = 1000
BATCH_SIZE = 1000
data_path = 'data/spectrograms/topcoder/trainingdata/sox'
data_ids_labels_path = 'data/raw/topcoder/trainingData.csv'


# 2. Define some functions/classes:
class ImagesBuffer(BufferThread):
    def loader(self, source_path: str, *args, **kwargs):
        i = imageio.imread(source_path)
        return np.asarray(i / 255)


def img_load(*a, **k):
    i = imageio.imread(*a, **k)
    # i = i.transpose((1, 0, 2))
    # i = np.flip(i, 1)
    return np.asarray(i/255)


def image_props(batch):
    """Returns the width and height of a batch of images"""
    return batch.shape[1], batch.shape[2]


# 3. Main:
if __name__ == '__main__':

    # 3.1. Imports:
    from keras import backend as K
    from keras import optimizers
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras.utils import to_categorical
    from keras.utils.vis_utils import plot_model
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    # 3.2. Model constants:
    batch_norm = True
    dropout = 0.5
    epochs = 100

    # 3.3. Set the data
    # 3.3.1. Create a buffer to store the data:
    images = queue.Queue(BUFFER_SIZE)
    data_ids_labels = open(data_ids_labels_path, 'r')
    buffer_thread = ImagesBuffer(data_ids_labels=data_ids_labels.
                                 readlines()[1:],
                                 source=data_path,
                                 buffer=images)
    buffer_thread()
    names, specs = buffer_thread.get_batch(BATCH_SIZE)
    buffer_thread.stop()

    print('[INFO] setting up the data set')

    # 3.3.2. Set up the data set:
    data_ids_labels = open(data_ids_labels_path, 'r').readlines()[1:]
    ids = []
    labels = []
    for line in data_ids_labels:
        ids.append(line.split(',')[0])
        labels.append(line.split(',')[1][:-2])

    X = []
    y = []
    # Slow part: think in a better way to do it
    for i in range(len(data_ids_labels)):
        for j in range(len(names)):
            if names[j] == ids[i][:-4]:
                X.append(specs[j])
                y.append(labels[j])
    print('DONE\n')
    print('[INFO] splitting data')

    # 3.4. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = np.asarray(X_train), np.asarray(X_test),\
                                       np.asarray(y_train), np.asarray(y_test)

    labels = np.unique(labels)
    num_classes = labels.shape[0]
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    y_test = le.transform(y_test)
    y_train = le.transform(y_train)

    y_test = to_categorical(y_test, num_classes=num_classes)
    y_train = to_categorical(y_train, num_classes=num_classes)

    print('DONE\n')

    print('Size of train data set: {}, size of test data set: {}'.
          format(X_train.shape[0], X_test.shape[0]))

    print('Shape of X train data set: {}, shape of y train data set: {}'.
          format(X_train.shape, y_train.shape))

    print('Shape of X test data set: {}, shape of y test data set: {}'.
          format(X_test.shape, y_test.shape))

    width, height = image_props(specs)

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, height, width)
        X_test = X_test.reshape(X_test.shape[0], 1, height, width)
        input_shape = 1, height, width
    else:
        X_train = X_train.reshape(X_train.shape[0], height, width, 1)
        X_test = X_test.reshape(X_test.shape[0], height, width, 1)
        input_shape = height, width, 1

    # 3.5. Network modeling:
    def build(input_shape, classes, optimizer,
              dropout_rate=0, batch_normalization=False):

        m = Sequential()

        # 3.5.1. Convolutional layers:

        # 3.5.1.1. Convolutional layer 1
        m.add(Conv2D(filters=16, kernel_size=(7, 7), strides=1,
                         activation='relu',
                         input_shape=input_shape))
        m.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        if batch_normalization:
            m.add(BatchNormalization())

        # 3.5.1.2. Convolutional layer 2
        m.add(Conv2D(filters=32, kernel_size=(5, 5), strides=1,
                         activation='relu'))
        m.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        if batch_normalization:
            m.add(BatchNormalization())

        # 3.5.1.3. Convolutional layer 3
        m.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                         activation='relu'))
        m.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        if batch_normalization:
            m.add(BatchNormalization())

        # 3.5.1.4. Convolutional layer 4
        m.add(Conv2D(128, (3, 3), strides=1, activation='relu'))
        m.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        if batch_normalization:
            m.add(BatchNormalization())

        # 3.5.1.5. Convolutional layer 5
        m.add(Conv2D(128, (3, 3), strides=1, activation='relu'))
        m.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        if batch_normalization:
            m.add(BatchNormalization())

        # 3.5.1.6. Convolutional layer 6
        m.add(Conv2D(256, (3, 3), strides=1, activation='relu'))
        m.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        if batch_normalization:
            m.add(BatchNormalization())

        # 3.5.2. Dense layers:

        m.add(Flatten())
        m.add(Dense(1024, activation='relu'))
        if batch_normalization:
            m.add(BatchNormalization())

        if dropout_rate > 0:
            m.add(Dropout(dropout_rate))

        # 3.5.3. Classification layer:

        m.add(Dense(classes, activation='softmax'))

        m.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return m

    sgd = optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    model = build(input_shape=input_shape, classes=num_classes, optimizer=sgd)

    print(model.summary())

    plot_model(model, to_file='models/scratch_model.png', show_shapes=True,
               show_layer_names=True)

    # 3.6. Network training:

    # H = model.fit_generator # todo
    H = model.fit(X_train, y_train, batch_size=8, epochs=epochs, verbose=1,
                  validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('[INFO] serializing network...')
    model.save('models/scratch_model')
    print('DONE')

    # 3.7. Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('scratch_train.png')
