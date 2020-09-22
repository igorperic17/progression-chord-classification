
import string
import numpy as np

# single audio sample data and label

class AudioSample:

    def __init__(self, label:string, data):
        self.label = label
        self.data = data
    
    # returns scaled copy of the data sample
    def get_scaled(self, factor:float):
        new_sample = self.data.copy()
        new_sample *= factor
        return AudioSample(self.label, new_sample)

    # return a copy shifted in time
    # factor is the index offset (not time)
    # to compute this offset take into account the samplerate
    def get_shifted(self, factor:int):
        
        new_sample = self.data.copy()

        start = 0
        end = len(self.data) - abs(factor)
        
        if factor > 0:
            for i in range(start, end):
                new_sample[i + factor] = new_sample[i]
        else:
            for i in range(start, end):
                new_sample[i] = new_sample[i - factor]
        
        return AudioSample(self.label, new_sample)
    
    def get_noisy(self, amount:float):
        new_sample = self.data.copy()
        noise = np.random.normal(0, amount, len(self.data)).tolist()
        new_sample += noise
        return AudioSample(self.label, new_sample)
