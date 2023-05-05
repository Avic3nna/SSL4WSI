__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"

import os
import json
import time
import torch
import matplotlib.pyplot as plt

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from monai.config import print_config
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
)

import argparse
from pathlib import Path
import PIL
from helpers import init_train_transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalise WSI directly.')

    parser.add_argument('-d', '--data', type=Path, required=True,
                        help='Path to load data from.')
    parser.add_argument('--wsi-dir', metavar='DIR', type=Path, required=True,
                        help='Path of where the whole-slide images are.')
    parser.add_argument('-m', '--model', metavar='DIR', type=Path, required=True,
                        help='Path of where model for the feature extractor is.')

PIL.Image.MAX_IMAGE_PIXELS = None

if __name__ == "__main__":
    print(args.

    set_determinism(seed=1337)
