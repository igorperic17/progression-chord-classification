
from data_parser import *
import models.AudioSample

def test1():
    audio_data: AudioSample = parse_audio_file('raw_data/Am.wav')
    sample_time_horizon = None
    for sample in audio_data:
        assert(sample.label == 'Am')
        assert(sample.data.all() != None)
        if sample_time_horizon is None:
            sample_time_horizon = len(sample.data)
        print(str(len(sample.data)) + ', ' + str(sample_time_horizon))
        assert(len(sample.data) == sample_time_horizon)
    return True

# run tests
# test1()