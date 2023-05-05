__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"

import os
import json
import time
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import PIL

from monai.utils import set_determinism
from monai.data import CacheDataset

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from helpers.training_transforms import ssl_transforms
from helpers.config_ssl import SSLModel
from helpers.load_data import load_tile_sets
from helpers.utils import collate_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SSL for WSI using ViT.')

    parser.add_argument('-d', '--data_dir', type=Path, required=True,
                        help='Path to load data from.')
    parser.add_argument('-l', '--log_dir', type=Path, required=True,
                        help='Path of where to log the output.')
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                        help='Batch size of the data loader.')

    args = parser.parse_args()


PIL.Image.MAX_IMAGE_PIXELS = None


if __name__ == "__main__":
    set_determinism(seed=1337)
    train_data, val_data = load_tile_sets(tile_path=args.data_dir)
    # Define DataLoader using MONAI, CacheDataset needs to be used
    train_ds = CacheDataset(data=train_data, transform=ssl_transforms, cache_rate=1., num_workers=8)
    train_sampler = DistributedSampler(train_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    val_ds = CacheDataset(data=val_data, transform=ssl_transforms, cache_rate=1., num_workers=8)
    val_sampler = DistributedSampler(val_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    val_loader = DataLoader(val_ds, sampler=val_sampler, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
    model = SSLModel()
    model.train(train_loader=train_loader, val_loader=val_loader, output_path=args.log_dir)
    model.plot(output_path=args.log_dir)
