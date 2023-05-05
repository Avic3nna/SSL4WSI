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
from helpers.training_transforms import ssl_transforms
from helpers.config_ssl import SSLModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Normalise WSI directly.')

    parser.add_argument('-d', '--data', type=Path, required=True,
                        help='Path to load data from.')
    parser.add_argument('-l', '--log_dir', type=Path, required=True,
                        help='Path of where to log the output.')
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                        help='Batch size of the data loader.')

    args = parser.parse_args()

PIL.Image.MAX_IMAGE_PIXELS = None

if __name__ == "__main__":
    set_determinism(seed=1337)
    # Define DataLoader using MONAI, CacheDataset needs to be used
    train_ds = Dataset(data=train_data, transform=ssl_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_ds = Dataset(data=val_data, transform=ssl_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = SSLModel()
    model.train(train_loader=train_loader, val_loader=val_loader, output_path=args.log_dir)
    model.plot(output_path=args.log_dir)
