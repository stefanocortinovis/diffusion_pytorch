"""
Reproduce experiments on CIFAR-10 for NCSN++ 
"""
from configs.ve.cifar10_ncsnpp import get_config
from samples import sample

from pathlib import Path

import torch


# DDPM paths and configuration file for CIFAR10
config = get_config()
checkpoint_path = Path('./experiments/cifar10_ncsnpp_ve/checkpoints/checkpoint_16.pth')
sample_dir = Path('./experiments/cifar10_ncsnpp_ve/samples')

sample(config, checkpoint_path, sample_dir, n=100, batch_size=1)
