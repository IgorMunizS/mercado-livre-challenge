import numpy as np

import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, Y=None, classes=None, batch_size=32, dim=(32, 32, 32),
                 shuffle=True):
        'Initialization'
        self.X = X
        self.Y = Y
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.classes = classes
        self.n_classes = len(self.classes)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        X_temp = [self.X[k] for k in indexes]
        Y_temp = [self.Y[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(X_temp, Y_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, X, Y):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.empty((self.batch_size), dtype=int)
        for i in range(len(Y)):
            y[i] = list(self.classes).index(Y[i])

        x = np.array(X)

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)