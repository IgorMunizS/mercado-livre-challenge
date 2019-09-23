import numpy as np

import keras
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import AllKNN

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, Y=None, classes=None, batch_size=32, dim=(32, 32, 32),
                 shuffle=True, mode='normal', train=True):
        'Initialization'
        self.mode = mode
        if self.mode == 'normal':
            self.X = X
        else:
            self.X = X[0]
            self.X_2 = X[1]
            # self.X_3 = X[2]

        self.Y = Y
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train

        self.resample = AllKNN(random_state=42, n_jobs=42)

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
            # X_temp_3 = [self.X_3[k] for k in indexes]

            X_temp = [X_temp, X_temp_2]

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

        if self.train:
            if self.mode == 'three':
                X_res = np.concatenate((X[0], X[1]), axis=1)

                X_res, y = self.resample.fit_resample(X_res, y)


                X = [X_res[:,:20],X_res[:,20:]]
            else:

                X, y = self.resample.fit_resample(X, y)

        else:
            if self.mode == 'three':
                X = [np.array(X[0]), np.array(X[1])]
            else:
                X = np.array(X)




        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#
#     def __init__(self, X, Y=None, classes=None, batch_size=32, dim=(32, 32, 32),
#                  shuffle=True, mode='normal'):
#         'Initialization'
#         self.mode = mode
#         if self.mode == 'normal':
#             self.X = X
#             self.n_inputs = 1
#         else:
#             self.X = X
#             self.n_inputs = len(self.X)
#
#
#         self.Y = Y
#         self.dim = dim
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#
#         self.classes = classes
#         self.n_classes = len(self.classes)
#
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.X) / self.batch_size))
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#         # Find list of IDs
#         if self.mode == 'normal':
#             X_temp = [self.X[k] for k in indexes]
#
#         else:
#             X_temp = []
#             for i in range(len(self.X)):
#                 x_temp = [self.X[i][k] for k in indexes]
#                 X_temp.append(x_temp)
#
#
#         Y_temp = [self.Y[k] for k in indexes]
#         # Generate data
#         X, y = self.__data_generation(X_temp, Y_temp)
#
#         return X, y
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#
#         if self.mode == 'three':
#             self.indexes = np.arange(len(self.X[0]))
#         else:
#             self.indexes = np.arange(len(self.X))
#
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, x, Y):
#         'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
#         # Initialization
#         y = np.empty((self.batch_size), dtype=int)
#         for i in range(len(Y)):
#             y[i] = list(self.classes).index(Y[i])
#
#
#         if self.mode == 'three':
#             X =[]
#             for i in range(len(x)):
#                 X.append(np.array(x[i]))
#
#         else:
#             X = np.array(x)
#
#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)