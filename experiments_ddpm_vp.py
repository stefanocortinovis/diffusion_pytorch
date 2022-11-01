"""
Reproduce experiments on CIFAR-10 for DDPM 
"""
from configs.vp.ddpm.cifar10 import get_config as get_config_ddpm_cifar10
from samples import sample

from pathlib import Path


# DDPM paths and configuration file for CIFAR10
config_ddpm = get_config_ddpm_cifar10()
checkpoint_ddpm = Path('./experiments/cifar10_ddpm_vp/checkpoints/checkpoint_14.pth')
dir_ddpm = Path('./experiments/cifar10_ddpm_vp/samples')

sample(config_ddpm, checkpoint_ddpm, dir_ddpm, n=100, batch_size=100)
