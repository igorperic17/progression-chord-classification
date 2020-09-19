from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from models.ChordsDataset import ChordsDataset

import random
import numpy as np

class ChordDetectionPipeline:

    def __init__(self):

        # storage for the result of the training process
        # after training it can be used to generate predictions and validation
        self.trained_model = None

        self.train_data = None
        self.train_labels = None
        self.validation_data = None
        self.validation_labels = None

        self.pipeline: Sequential = None

        self.dataset: ChordsDataset = None

    @classmethod
    def create_with_dataset(cls, dataset: ChordsDataset):
        obj = cls()
        obj.dataset = dataset
        return obj

    # private method, prepares the keras model
    def create_network(self):

        feature_vector_size = self.dataset.feature_vector_size
        output_vector_size = len(self.dataset.chord_labels)

        self.pipeline = Sequential()
        self.pipeline.add(Conv1D(16, kernel_size=100, input_shape=(feature_vector_size, 1), padding='same',activation='relu'))
        self.pipeline.add(Flatten())
        self.pipeline.add(Dense(output_vector_size, activation='softmax'))

        self.pipeline.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])

        print(self.pipeline.summary())

    # start the training of the model
    def train(self, train_valid_split_ratio = 0.2):

        # perform balanced splits of all of the chord classes on the raw data
        # and construct properly shaped numpy inputs
        self.prepare_input(train_valid_split_ratio)

        # create keras NN model
        self.create_network()

        # perform model fitting
        self.pipeline.fit(self.train_data, self.train_labels, \
            epochs=10, verbose=1, validation_data=(self.validation_data, self.validation_labels))

    def predict(self):

        if self.trained_model is None:
            print('Model is not trained, prediction can\'t be made. Aborting.')
            return None
        
        # TODO: make prediction
        return None
    
    # Performes balanced train/validation split.
    def prepare_input(self, train_valid_split_ratio = 0.2):

        self.train_data = np.empty(shape=(0,self.dataset.feature_vector_size,1), dtype=float)
        self.train_labels = np.empty(shape=(0,len(self.dataset.chord_labels)), dtype=float)
        self.validation_data = np.empty(shape=(0,self.dataset.feature_vector_size, 1), dtype=float)
        self.validation_labels = np.empty(shape=(0,len(self.dataset.chord_labels)), dtype=float)

        # for each label in our dataset...
        for idx, label in enumerate(self.dataset.chord_labels):
            print('Preparing input for label: ' + label)
            # ... isolate indices of AudioSample objects with the current label
            indices = [i for i, x in enumerate(self.dataset.samples) if x.label == label ]
            random.shuffle(indices)
            
            # determine the number of elements to take for this label
            n_train = int(len(indices) * (1.0 - train_valid_split_ratio))
            n_valid = len(indices) - n_train
            idx_train = indices[:n_train]
            idx_valid = indices[-n_valid:]

            selected_data = np.array(list(map(lambda i: self.dataset.samples[i].data, idx_train)))
            selected_data = np.reshape(selected_data, (-1, selected_data.shape[1], 1))
            self.train_data = np.vstack((self.train_data, selected_data))
            selected_data = np.array(list(map(lambda i: self.dataset.samples[i].data, idx_valid)))
            selected_data = np.reshape(selected_data, (-1, selected_data.shape[1], 1))
            self.validation_data = np.vstack((self.validation_data, selected_data))

            selected_data = np.array(list(map(lambda i: self.dataset.samples[i].label, idx_train)))
            # convert labels to one-hot encoding
            selected_data = np.array([self.dataset.one_hot_encoding(x) for x in selected_data])
            # reshape to proper number of columns
            selected_data = np.reshape(selected_data, (-1, len(self.dataset.chord_labels)))
            self.train_labels = np.vstack((self.train_labels, selected_data))
            
            selected_data = np.array(list(map(lambda i: self.dataset.samples[i].label, idx_valid)))
            # convert labels to one-hot encoding
            selected_data = np.array([self.dataset.one_hot_encoding(x) for x in selected_data])
            # reshape to proper number of columns
            selected_data = np.reshape(selected_data, (-1, len(self.dataset.chord_labels)))
            self.validation_labels = np.vstack((self.validation_labels, selected_data))

        print(self.train_data.shape)
        print(self.train_labels.shape)
        print(self.validation_data.shape)
        print(self.validation_labels.shape)
