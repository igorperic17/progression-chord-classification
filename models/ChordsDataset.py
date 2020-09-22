

import string, os
from models.AudioSample import *

import data_parser

class ChordsDataset:

    def __init__(self):
        self.chord_labels: [string] = []
        self.samples: [AudioSample] = []
        self.sample_time_horizon: float = None # time horizon length in seconds
        self.feature_vector_size: int = None # sample_time_horizon * audio_sampling_rate

    @classmethod
    def create_from_samples(cls, samples: [AudioSample]):
        obj = cls()
        obj.samples = samples
        # TODO: construct the self.chord_labels based on samples.

        return obj

    # Constructs a ChordsDataset object, which is a complete list of AudioSample objects 
    # for all of the raw audio files at the provided root path.
    # The object also contains the list of unique labels.
    @classmethod
    def create_from_path(cls, path:string, sample_time_horizon:float):
        obj = cls()
        obj.sample_time_horizon = sample_time_horizon

        files = os.listdir(path)
        for filename in files:
            if filename.endswith('.wav'):
                print('Processing file: ' + filename)
                obj.chord_labels.append(filename.replace('.wav',''))
                current_sample_list = data_parser.parse_audio_file(path + "/" + filename, sample_time_horizon)
                obj.samples += current_sample_list
        
        obj.chord_labels.sort()

        # sanity check - do all of the samples have the same size?
        obj.feature_vector_size = None
        for sample in obj.samples:
            assert(sample.data.all() != None)
            if obj.feature_vector_size is None:
                obj.feature_vector_size = len(sample.data)
            assert(len(sample.data) == obj.feature_vector_size)

        print('Created a dataset with ' + str(len(obj.samples)) + ' chord samples.')

        return obj

    def one_hot_encoding(self, label:string):
        encoding = [0] * len(self.chord_labels)
        encoding[self.chord_labels == label] = 1
        return encoding

    def perform_augmentations(self):

        new_samples: [AudioSample] = []
        for i in range(len(self.samples)):

            sample = self.samples[i]

            print('Performing augmentation of sample ' + str(i) + ' out of ' + str(len(self.samples)))

            # magnitude scaling
            new_samples.append(sample.get_scaled(1.4))
            new_samples.append(sample.get_scaled(1.2))
            new_samples.append(sample.get_scaled(0.8))
            new_samples.append(sample.get_scaled(0.6))

            # time shift
            new_samples.append(sample.get_shifted(44)) # 0.1 second offset for 44.1 kHz
            new_samples.append(sample.get_shifted(-44)) # 0.1 second offset for 44.1 kHz
            new_samples.append(sample.get_shifted(88)) # 0.1 second offset for 44.1 kHz
            new_samples.append(sample.get_shifted(-88)) # 0.1 second offset for 44.1 kHz
            new_samples.append(sample.get_shifted(200)) # 0.1 second offset for 44.1 kHz
            new_samples.append(sample.get_shifted(-200)) # 0.1 second offset for 44.1 kHz

            # white additive noise
            new_samples.append(sample.get_noisy(0.5))
            new_samples.append(sample.get_noisy(1))
            new_samples.append(sample.get_noisy(2))
        
        self.samples += new_samples