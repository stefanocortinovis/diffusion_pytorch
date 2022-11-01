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
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

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

        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape((
            -1,
            config.data.image_size,
            config.data.image_size,
            config.data.num_channels
        ))

        # Write samples to disk
        with tf.io.gfile.GFile(os.path.join(sample_dir, f"samples_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())

if __name__ == '__main__':
    # DDPM paths and configuration file for CIFAR10
    dir_ddpm = Path('./samples/ddpm')
    checkpoint_ddpm = Path('./exp/vp/cifar10_ddpm/checkpoint_14.pth')
    config_ddpm = get_config_ddpm_cifar10()

    sample(config_ddpm, checkpoint_ddpm, dir_ddpm, n=100, batch_size=100)
