

from models.ChordsDataset import *
from models.ChordDetectionPipeline import *

dataset = ChordsDataset.create_from_path('raw_data', sample_time_horizon=1.0)

pipeline = ChordDetectionPipeline.create_with_dataset(dataset)
model = pipeline.train()

# test1()