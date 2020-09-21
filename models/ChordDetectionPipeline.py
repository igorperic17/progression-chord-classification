from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow

from models.ChordsDataset import ChordsDataset

import random, string
import numpy as np

import coremltools

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
        self.pipeline.add(Input(shape=(feature_vector_size, 1), name='waveform'))
        self.pipeline.add(Dropout(0.4))
        self.pipeline.add(Conv1D(3, kernel_size=256, padding='same', activation='relu'))
        self.pipeline.add(Dropout(0.4))
        # self.pipeline.add(Conv1D(64, kernel_size=32, padding='same', activation='relu'))
        # self.pipeline.add(Dropout(0.4))
        self.pipeline.add(BatchNormalization())
        self.pipeline.add(Flatten())
        # self.pipeline.add(Dense(32,activation=tensorflow.nn.relu))
        # self.pipeline.add(Dense(3,activation=tensorflow.nn.relu))
        self.pipeline.add(Dense(output_vector_size, name='chord_label', activation=tensorflow.nn.softmax))

        self.pipeline.compile(optimizer=Adam(0.01), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        print(self.pipeline.summary())

    # start the training of the model
    def train(self, train_valid_split_ratio = 0.2, epochs = 10):

        # perform balanced splits of all of the chord classes on the raw data
        # and construct properly shaped numpy inputs
        self.prepare_input(train_valid_split_ratio)

        # create keras NN model
        self.create_network()

        # perform model fitting
        self.pipeline.fit(self.train_data, self.train_labels, \
            epochs=epochs, verbose=1, validation_data=(self.validation_data, self.validation_labels))

        return self.pipeline

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

    # saves the model as .h5 and .mlmodel for use in iOS
    def save(self, model_name:string):
        
        self.pipeline.save(model_name + '.h5', overwrite=True, include_optimizer=True)
        # m = tensorflow.keras.models.load_model(model_name + '.h5')
        # m = tensorflow.tool
        
        output_labels = self.dataset.chord_labels
        
        m = coremltools.convert(self.pipeline, \
            input_names=['waveform'], output_names=['chord_label'], \
                class_labels=output_labels, source='tensorflow')

        m.author = 'Igor Peric'
        m.short_description = 'Chord classification based on raw audio data stream.'
        m.input_description['waveform'] = 'Takes as input 1D waveform, 44.1 kHz, 16-bit PCM signed. 1 second of data, vector of size 44100 elements.'
        # model.output_description['chord_name'] = 'Prediction of the chord name.'
        
        m.save(model_name + '.mlmodel')

