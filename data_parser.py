
import string
import pandas as pd
import os
from scipy.io import wavfile
import numpy as np
import csv

# models
from models.AudioSample import *
from models.ChordsDataset import * 

# Parses a single .waw file containing multiple audio samples.
# Labels are expected to be found in <wav_file_name>-labels.txt file.
# sample_time_horizon is the length of the sample in seconds starting from the onset.
# Returns a list of AudioSample objects containing labels and actual audio data,
# all of this for a single provided .wav file.
def parse_audio_file(path:string, sample_time_horizon, subsampling_skip_count=1):

    # extract the file name from the provided full path
    filename = os.path.basename(path)
    chord_label = filename.replace('.wav','')
    labels_filename = path.replace('.wav', '.txt')
    
    print('Opening labels file: ' + labels_filename)
    if not os.path.exists(labels_filename):
        print('Labels file not found for provided .wav file. Aborting.')
        return None

    data_samples = [] # list of all samples defined by the labels .txt fale
    samplerate, audio_data = wavfile.read(path) # read the whole audio...
    # audio_data = audio_data[:, 1] # take only single channel, since we're getting stereo data
    print(audio_data[:100])
    # read labels from the .txt file, row by row
    with open(labels_filename, 'r') as labels_file:
        sample_labels = csv.reader(labels_file, delimiter='\t')
        for label in sample_labels:
            # extract the audio segment
            start_sec = float(label[0])
            start_pos = int(start_sec * samplerate)
            # end_sec = float(label[1])
            end_pos = start_pos + int(sample_time_horizon * samplerate) # fixed size horizon
            # print(str(start_pos) + ', ' + str(end_pos))
            current_sample_data = audio_data[start_pos:end_pos:subsampling_skip_count]
            # print(current_sample_data)
            data_sample = AudioSample(chord_label, current_sample_data)
            data_samples.append(data_sample)

    # sanity check - make sure all of the samples are of the same size
    target_sample_time_horizon = None
    for sample in data_samples:
        assert(sample.label == chord_label)
        assert(sample.data.all() != None)
        if target_sample_time_horizon is None:
            target_sample_time_horizon = len(sample.data)
        assert(len(sample.data) == target_sample_time_horizon)

    return data_samples





