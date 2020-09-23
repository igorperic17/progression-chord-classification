

from models.ChordsDataset import *
from models.ChordDetectionPipeline import *

dataset = ChordsDataset.create_from_path('raw_data', sample_time_horizon=1.0)
# dataset.perform_augmentations()

pipeline = ChordDetectionPipeline.create_with_dataset(dataset)
pipeline.train(train_valid_split_ratio=0.3, epochs=10)

pipeline.save('ChordDetectionModel')

# test1()