from data_parser import *
from tests.data_parser_tests import *

from models.ChordsDataset import *

dataset = ChordsDataset.create_from_path('raw_data')

# test1()