"""
Reproduce experiments on CIFAR-10 for DDPM and NCSN
"""

import datasets
import losses
import sampling
import sde_lib
from configs.vp.ddpm.cifar10 import get_config as get_config_ddpm_cifar10
from models import utils as mutils
from models import ddpm
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint

import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

def sample(config, checkpoint_file, sample_dir, n=100, batch_size=100):
    config.seed = 42

    # Set number of samples and batch size
    config.eval.num_samples = n
    config.eval.batch_size = batch_size

    # Create inverse scaler
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Setup SDEs
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3


    # Build the sampling function when sampling is enabled
    sampling_shape = (
        config.eval.batch_size,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size
    )
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, save_all=True)

    # Restore checkpoint
    state = restore_checkpoint(checkpoint_file, state, device=config.device)
    ema.copy_to(score_model.parameters()) # TODO: check where used

    # Compute number of sampling rounds
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + ((config.eval.num_samples % config.eval.batch_size) > 0)

    # Directory to save samples
    tf.io.gfile.makedirs(sample_dir)

    # Sample
    for r in range(num_sampling_rounds):
        print(f"Sampling round: {r + 1}/{num_sampling_rounds}")

        _, all_samples, _ = sampling_fn(score_model)
        print(all_samples.shape)
        all_samples = np.clip(all_samples.transpose((1, 0, 3, 4, 2)) * 255., 0, 255).astype(np.uint8)
        all_samples = all_samples.reshape((
            -1,
            sde.N + 1,
            config.data.image_size,
            config.data.image_size,
            config.data.num_channels
        ))

        # Write samples to disk
        with tf.io.gfile.GFile(os.path.join(sample_dir, f"samples_batch_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=all_samples)
            fout.write(io_buffer.getvalue())

if __name__ == '__main__':
    # DDPM paths and configuration file for CIFAR10
    config_ddpm = get_config_ddpm_cifar10()
    checkpoint_ddpm = Path('./experiments/cifar10_ddpm/checkpoints/checkpoint_14.pth')
    dir_ddpm = Path('./experiments/cifar10_ddpm/samples')

    sample(config_ddpm, checkpoint_ddpm, dir_ddpm, n=100, batch_size=100)
