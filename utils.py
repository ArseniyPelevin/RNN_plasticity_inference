"""Module with utility functions, taken as is from https://github.com/yashsmehta/MetaLearnPlasticity"""

import logging
import pickle
from itertools import count
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import pandas as pd


def setup_platform(device: str) -> None:
    """Set up the environment based on the configuration."""

    # Suppress JAX backend initialization messages if using CPU
    if device == "cpu":
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
    elif device == "gpu":
        try:
            jax.config.update("jax_platform_name", device)
        except Exception as e:
            logging.warning(f"Could not set JAX platform to {device}: {e}")

    device = jax.lib.xla_bridge.get_backend().platform
    logging.info(f"Platform: {device}\n")

def softclip(x, cap, p=10):
    return x / ((1.0 + jnp.abs(x / cap) ** p) ** (1.0 / p))

def sample_truncated_normal(key, mean, std, shape=1):
    """ Samples values from a normal distribution that are >= (mean - std). """
    samples = jax.random.normal(key, shape)
    while not jnp.all(samples >= -1):
        key, subkey = jax.random.split(key)
        samples2 = jax.random.normal(subkey, shape)
        samples = jnp.where(samples < -1, samples2, samples)
    samples = samples * std + mean
    return samples.round().astype(jnp.int32)

def print_and_log_learned_params(cfg, expdata, thetas):
    """
    Logs and prints current plasticity coefficients.

    Args:
        cfg (object): Configuration object containing the model settings.
        expdata (dict): Dictionary to store experimental data.
        thetas (dict): per-plastic_layer dict of
            (3, 3, 3, 3) arrays of plasticity coefficients.

    Returns:
        dict: Updated experimental data dictionary.
    """
    def print_and_log_learned_params_layer(layer, theta):
        coeff_prefix = layer[0].upper()  # 'F': feedforward, 'R': recurrent, 'B': both
        if cfg.plasticity.plasticity_models[layer] == "volterra":
            coeff_mask = jnp.array(cfg.plasticity.coeff_masks[layer])
            theta = jnp.multiply(theta, coeff_mask)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            dict_key = f"{coeff_prefix}_{i}{j}{k}{l}"
                            expdata.setdefault(dict_key, []).append(
                                theta[i, j, k, l]
                            )

            ind_i, ind_j, ind_k, ind_l = coeff_mask.nonzero()
            top_indices = jnp.argsort(
                jnp.abs(theta[ind_i, ind_j, ind_k, ind_l].ravel())
            )[-5:]

            print(f"\nTop learned plasticity terms for {layer} layer:")
            print("{:<10} {:<20}".format("Term", "Coefficient"))

            for idx in reversed(top_indices):
                term_str = ""
                if ind_i[idx] == 1:
                    term_str += "X "
                elif ind_i[idx] == 2:
                    term_str += "X² "
                if ind_j[idx] == 1:
                    term_str += "Y "
                elif ind_j[idx] == 2:
                    term_str += "Y² "
                if ind_k[idx] == 1:
                    term_str += "W "
                elif ind_k[idx] == 2:
                    term_str += "W² "
                if ind_l[idx] == 1:
                    term_str += "R"
                elif ind_l[idx] == 2:
                    term_str += "R²"
                coeff = theta[ind_i[idx], ind_j[idx], ind_k[idx], ind_l[idx]]
                print(f"{term_str:<10} {coeff:<20.5f}")
        else:
            print(f"MLP plasticity coeffs for {layer} layer: ", theta)
            expdata.setdefault("mlp_params", []).append(theta)

    for layer, theta in thetas.items():
        print_and_log_learned_params_layer(layer, theta)

    return expdata

def save_results(cfg, params, expdata, train_time, trajectories,
                 train_experiments, test_experiments):
    """Save training logs and parameters."""

    def create_directory():
        path = Path(cfg.logging.log_dir)
        path.mkdir(parents=True, exist_ok=True)  # Create log_dir if it doesn't exist
        dir_name = f"Exp_{exp_id}"
        for i in count():
            name = dir_name if i == 0 else f"{dir_name}_({i})"
            candidate = path / name
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        return candidate.as_posix() + '/'

    exp_id = cfg.logging.exp_id

    path = create_directory()

    # Save configuration
    if cfg.logging.log_config:
        # Save configuration used for the experiment
        omegaconf.OmegaConf.save(cfg, path + f"Exp_{exp_id}_config.yaml")

    # Save final parameters
    if cfg.logging.log_final_params:
        with open(path + f"Exp_{exp_id}_final_params.pkl", "wb+") as f:
            pickle.dump(jax.device_get(params), f)

    if cfg.logging.log_expdata:
        # Save expdata as .csv
        df = pd.DataFrame.from_dict(expdata)
        df["train_time"] = train_time
        df.to_csv(path + f"Exp_{exp_id}_results.csv", mode="w+", index=False)
        # TODO allow appending to existing file if retraining

    if cfg.logging.log_generated_experiments:
        # Save generated experiments
        for name, experiments in zip(["train", "test"],
                                     [train_experiments, test_experiments],
                                     strict=False):
            exp_path = Path(path + f"generated_{name}_experiments")
            exp_path.mkdir(parents=True, exist_ok=True)
            for i, exp in enumerate(experiments):
                eqx.tree_serialise_leaves(exp_path / f"experiment_{i}.eqx", exp)

    if cfg.logging.log_trajectories:
        # Save trajectories
        save_nested_hdf5(trajectories, path + f"Exp_{exp_id}_trajectories.h5")

    return path

# Older version allowing appending to existing log files
def save_logs(cfg, df):
    """
    Saves the logs to a specified directory based on the configuration.

    Args:
        cfg (object): Configuration object containing the model settings and paths.
        df (DataFrame): DataFrame containing the logs to be saved.

    Returns:
        Path: The path where the logs were saved.
    """

    # local_random = random.Random()
    # local_random.seed(os.urandom(10))
    # sleep_duration = local_random.uniform(1, 5)
    # time.sleep(sleep_duration)
    # print(f"Slept for {sleep_duration:.2f} seconds.")

    logdata_path = Path(cfg.log_dir)
    if cfg.log_expdata:

        logdata_path.mkdir(parents=True, exist_ok=True)
        csv_file = logdata_path / f"exp_{cfg.logging.exp_id}.csv"
        write_header = not csv_file.exists()

        lock_file = csv_file.with_suffix(".lock")
        while lock_file.exists():
            print(f"Waiting for lock on {csv_file}...")
            # time.sleep(random.uniform(1, 5))

        try:
            lock_file.touch()
            df.to_csv(csv_file, mode="a", header=write_header, index=False)
            print(f"Saved logs to {csv_file}")
        finally:
            lock_file.unlink()

    return logdata_path

def save_nested_hdf5(obj, path):
    """Save a nested dict of arrays to HDF5, preserving the tree."""
    with h5py.File(path, 'w') as f:
        for k, lst in obj.items():
            g = f.create_group(str(k))
            for i, rec in enumerate(lst):
                s = g.create_group(str(i))
                for name, arr in rec.items():
                    if name == 'weights':
                        wg = s.create_group('weights')
                        for wk, wv in arr.items():
                            wg.create_dataset(wk, data=np.array(wv))
                    else:
                        s.create_dataset(name, data=np.array(arr))

def load_nested_hdf5(fname):
    """Load HDF5 into nested dict (datasets -> numpy arrays)."""
    def _recursively_read(h):
        out = {}
        for k in h:
            item = h[k]
            if isinstance(item, h5py.Group):
                out[k] = _recursively_read(item)
            else:
                out[k] = item[()]  # read dataset into numpy array
        return out
    with h5py.File(fname, "r") as f:
        return _recursively_read(f)
