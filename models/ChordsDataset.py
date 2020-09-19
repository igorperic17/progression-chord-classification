

import string, os
from models.AudioSample import *

import data_parser

# single audio sample data and label

class ChordsDataset:

    def __init__(self):
        self.chord_labels: [string] = []
        self.samples: [AudioSample] = []

    @classmethod
    def create_from_samples(cls, samples: [AudioSample]):
        obj = cls()
        obj.samples = samples
        return obj

    # Constructs a ChordsDataset object, which is a complete list of AudioSample objects 
    # for all of the raw audio files at the provided root path.
    # The object also contains the list of unique labels.
    @classmethod
    def create_from_path(cls, path:string):
        obj = cls()

        files = os.listdir(path)
        for filename in files:
            if filename.endswith('.wav'):
                print('Processing file: ' + filename)
                obj.chord_labels.append(filename.replace('.wav',''))
                current_sample_list = data_parser.parse_audio_file(path + "/" + filename)
                obj.samples += current_sample_list
        
        # sanity check - do all of the samples have the same size?
        time_horizon = None
        for sample in obj.samples:
            assert(sample.data.all() != None)
            if time_horizon is None:
                time_horizon = len(sample.data)
            assert(len(sample.data) == time_horizon)

        print('Created a dataset with ' + str(len(obj.samples)) + ' chord samples.')
        return obj