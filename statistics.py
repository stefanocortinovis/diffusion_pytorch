import evaluation
from configs.vp.ddpm.cifar10 import get_config as get_config_ddpm_cifar10

import gc
import io
import os

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan


def statistics(config, samples_path, statistics_dir):
    # Set number of samples and batch size
    config.eval.num_samples = 100 # TODO: change
    config.eval.batch_size = 100 # TODO: change

    # Compute number of sampling rounds
    num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + ((config.eval.num_samples % config.eval.batch_size) > 0)

    # Load samples
    samples = np.load(samples_path)['samples'] # 100, 32, 32, 3 -> TODO: to be changed later

    # Load inception model
    inception_model = evaluation.get_inception_model() # note: inception v3 should be used if image resolution >= 256

    # Directory to save samples
    tf.io.gfile.makedirs(statistics_dir)

    for r in range(num_sampling_rounds):
        # Force garbage collection before calling TensorFlow code for Inception network
        gc.collect()

        latents = evaluation.run_inception(samples, inception_model)

        # Force garbage collection again before returning to JAX code
        gc.collect()

        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(statistics_dir, f"statistics_batch_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
            fout.write(io_buffer.getvalue())

    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []
    stats = tf.io.gfile.glob(os.path.join(statistics_dir, "statistics_*.npz"))
    for stat_file in stats:
        with tf.io.gfile.GFile(stat_file, "rb") as fin:
            stat = np.load(fin)
            all_logits.append(stat["logits"])
            all_pools.append(stat["pool_3"])
    all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]

    # Compute IS
    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)

    # Compute FID
    fid = tfgan.eval.frechet_classifier_distance_from_activations(data_pools, all_pools)

    # Compute KID with hack to get tfgan KID work for eager execution
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
    tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools

    with tf.io.gfile.GFile(os.path.join(statistics_dir, f"report.npz"), "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, IS=inception_score, FID=fid, KID=kid)
        f.write(io_buffer.getvalue())

if __name__ == '__main__':
    config_ddpm = get_config_ddpm_cifar10()
    samples_path = './experiments/cifar10_ddpm/samples/samples_0.npz'
    statistics_dir = './experiments/cifar10_ddpm/statistics'
    statistics(config_ddpm, samples_path, statistics_dir)
