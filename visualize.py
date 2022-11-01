import os

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def make_gif(samples_dir, n_rows):
    all_samples = []
    sample_files = tf.io.gfile.glob(os.path.join(samples_dir, "samples_batch_*.npz"))
    for sample_file in sample_files:
        with tf.io.gfile.GFile(sample_file, "rb") as fin:
            sample = np.load(fin)['samples']
            all_samples.append(sample)
    all_samples = torch.from_numpy(np.concatenate(all_samples, axis=0)[:n_rows ** 2])

    imgs = []
    for t in tqdm(range(all_samples.shape[1])):
        if t % 10 == 0:
            image_grid = make_grid(all_samples[:, t].permute(0, 3, 1, 2), nrow=n_rows)
            # image_grid = make_grid(all_samples[:, t].permute(0, 3, 1, 2), nrow=10, normalize=True, value_range=(0, 255))
            imgs.append(Image.fromarray(image_grid.permute(1, 2, 0).cpu().numpy()))
    imgs[0].save(os.path.join(samples_dir, f'samples_{n_rows ** 2}.gif'), save_all=True, append_images=imgs[1:], duration=1, loop=0)

