import numpy as np
import PIL
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split


def load_tile_sets(tile_path: Path = None):
    tile_dirs = tile_path.glob('*.jpg')
    train_set, valid_set = train_test_split(tile_dirs, test_size=0.2, random_state=1337)
    data_dict = {"training": [], "validation": []}
    
    # add the training files to the training dictionary
    for filename in train_set:
        data_dict["training"].append({"image": filename})

    # add the validation files to the validation dictionary
    for filename in valid_set:
        data_dict["validation"].append({"image": filename})

    return data_dict["training"], data_dict["validation"]
