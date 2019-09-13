import numpy as np

import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, Y=None, classes=None, batch_size=32, dim=(32, 32, 32),
                 shuffle=True, mode='normal'):
        'Initialization'
        self.mode = mode
        if self.mode == 'normal':
            self.X = X
        else:
            self.X = X[0]
            self.X_2 = X[1]
            self.X_3 = X[2]

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
        if self.mode == 'normal':
            X_temp = [self.X[k] for k in indexes]

        else:
            X_temp = [self.X[k] for k in indexes]
            X_temp_2 = [self.X_2[k] for k in indexes]
            X_temp_3 = [self.X_3[k] for k in indexes]

            X_temp = [X_temp, X_temp_2,X_temp_3]

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


        if self.mode == 'three':
            X = [np.array(X[0]),np.array(X[1]), np.array(X[2])]
        else:

            X = np.array(X)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)