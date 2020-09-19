
import string

# single audio sample data and label

class AudioSample:

    def __init__(self, label:string, data):
        self.label = label
        self.data = data
    