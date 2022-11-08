"""
Reproduce experiments on CIFAR-10 for DDPM 
"""
from configs.vp.cifar10_ddpmpp import get_config
from samples import sample
from visualize import make_gif

from pathlib import Path


# DDPM paths and configuration file for CIFAR10
config = get_config()
checkpoint_path = Path('./experiments/cifar10_ddpmpp_vp/checkpoints/checkpoint_6.pth')
samples_dir = Path('./experiments/cifar10_ddpmpp_vp/samples')

sample(config, checkpoint_path, samples_dir, n=100, batch_size=100)
make_gif(samples_dir, n_rows=10)
