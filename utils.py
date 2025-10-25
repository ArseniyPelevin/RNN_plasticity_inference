import ast
import glob
import json
import logging
import os
import pickle
from itertools import count
from pathlib import Path

import equinox as eqx
import experiment
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import pandas as pd


def setup_platform(device: str) -> None:
    """Set up the environment based on the configuration.
    Copied from https://github.com/yashsmehta/MetaLearnPlasticity"""

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
    Copied partially from https://github.com/yashsmehta/MetaLearnPlasticity

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
                 train_experiments=None, test_experiments=None, path=None):
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

    if not path:
        path = create_directory()

    # Save configuration
    if cfg.logging.log_config:
        # Save configuration used for the experiment
        save_config(cfg, path, exp_id)

    # Save final parameters
    if cfg.logging.log_final_params:
        save_final_params(params, path, exp_id)

    if cfg.logging.log_expdata:
        # Save expdata as .csv
        save_expdata(expdata, path, exp_id, train_time)

    if (cfg.logging.log_generated_experiments and
        train_experiments is not None and test_experiments is not None):
        # Save generated experiments
        save_generated_experiments(path, train_experiments, test_experiments)

    if cfg.logging.log_trajectories:
        # Save trajectories
        save_hdf5(trajectories, path + f"Exp_{exp_id}_trajectories.h5")

    return path

def load_old_experiment(path):
    """ Loads configuration, final parameters, expdata, generated experiments
    and trajectories from an older experiment saved in `path`. """

    cfg = load_config(path)
    params = load_final_params(path)
    expdata = load_expdata(path)
    train_experiments = load_generated_experiments(path, cfg, mode="train")
    test_experiments = load_generated_experiments(path, cfg, mode="test")
    trajectories = load_hdf5(path + f"Exp_{cfg.logging.exp_id}_trajectories.h5")

    return cfg, params, expdata, train_experiments, test_experiments, trajectories

def save_config(cfg, path, exp_id):
    """ Saves the configuration used for the experiment as a JSON file. """
    omegaconf.OmegaConf.save(cfg, path + f"Exp_{exp_id}_config.yaml")

def load_config(path):
    """ Loads the configuration used for the experiment from a YAML file. """
    path = glob.glob(os.path.join(path, "Exp_*_config.yaml"))[0]
    return omegaconf.OmegaConf.load(path)

def save_final_params(params, path, exp_id):
    """ Saves the final learned parameters as a pickle file. """
    with open(path + f"Exp_{exp_id}_final_params.pkl", "wb") as f:
        pickle.dump(params, f)

def load_final_params(path):
    """ Loads the final learned parameters from a pickle file. """
    path = glob.glob(os.path.join(path, "Exp_*_final_params.pkl"))[0]
    with open(path, "rb") as f:
        params = pickle.load(f)
    return params

def save_expdata(expdata, path, exp_id, train_time):
    df = pd.DataFrame.from_dict(expdata)
    df["train_time"] = train_time
    df.to_csv(path + f"Exp_{exp_id}_expdata.csv", mode="w+", index=False)
    # TODO allow appending to existing file if retraining

def load_expdata(path):
    path = glob.glob(os.path.join(path, "Exp_*_expdata.csv"))[0]
    df = pd.read_csv(path)
    expdata = df.to_dict(orient="list")
    return expdata

def save_generated_experiments(path, train_experiments, test_experiments):
    for name, experiments in zip(["train", "test"],
                                 [train_experiments, test_experiments],
                                 strict=False):
        exp_path = Path(path + f"generated_{name}_experiments")
        exp_path.mkdir(parents=True, exist_ok=True)
        for i, exp in enumerate(experiments):
            eqx.tree_serialise_leaves(exp_path / f"experiment_{i}.eqx", exp)

def load_generated_experiments(path, cfg, mode):
    exp_like = experiment.generate_experiments(jax.random.PRNGKey(0), cfg,
                                               mode=mode, num_exps=1)
    path += f"generated_{mode}_experiments/"
    # Number of files in path folder:
    num_files = len(list(Path(path).glob("experiment_*.eqx")))
    exps = []
    for exp_id in range(num_files):
        exp = eqx.tree_deserialise_leaves(path + f"experiment_{exp_id}.eqx",
                                          exp_like[0])
        exps.append(exp)

    return exps

def save_hdf5(obj, fname):
    with h5py.File(fname, "w") as f:
        def _write(g, o):
            if isinstance(o, dict):
                g.attrs["__type__"] = "dict"
                keys = list(o.keys())
                g.attrs["__keys__"] = json.dumps([[str(k), type(k).__name__]
                                                  for k in keys])
                for i, k in enumerate(keys):
                    sg = g.create_group(f"item_{i}")
                    _write(sg, o[k])
            elif isinstance(o, list):
                g.attrs["__type__"] = "list"
                g.attrs["__len__"] = len(o)
                for i, v in enumerate(o):
                    sg = g.create_group(f"item_{i}")
                    _write(sg, v)
            elif isinstance(o, tuple):
                g.attrs["__type__"] = "tuple"
                g.attrs["__len__"] = len(o)
                for i, v in enumerate(o):
                    sg = g.create_group(f"item_{i}")
                    _write(sg, v)
            else:
                # leaf: jax array
                g.attrs["__type__"] = "array"
                arr = np.asarray(jax.device_get(o))
                g.create_dataset("value", data=arr,
                                 compression="gzip", compression_opts=4)
        _write(f, obj)

def load_hdf5(fname):
    def _restore_key(s, tname):
        if tname == "int": return int(s)
        if tname == "float": return float(s)
        if tname == "bool": return s == "True"
        if tname == "str": return s
        if tname == "NoneType": return None
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    with h5py.File(fname, "r") as f:
        def _read(g):
            t = g.attrs.get("__type__", None)
            if t == b"dict": t = "dict"   # h5py may return bytes
            if t == b"list": t = "list"
            if t == b"tuple": t = "tuple"
            if t == b"array": t = "array"

            if t == "dict":
                keys_meta = json.loads(g.attrs["__keys__"])
                out = {}
                for i, (ks, tname) in enumerate(keys_meta):
                    out[_restore_key(ks, tname)] = _read(g[f"item_{i}"])
                return out
            if t == "list":
                n = int(g.attrs.get("__len__",
                                    len([k for k in g if k.startswith("item_")])))
                return [_read(g[f"item_{i}"]) for i in range(n)]
            if t == "tuple":
                n = int(g.attrs.get("__len__",
                                    len([k for k in g if k.startswith("item_")])))
                return tuple(_read(g[f"item_{i}"]) for i in range(n))
            if t == "array":
                data = g["value"][...]
                return jnp.array(data)
            # fallback (empty file/root without attrs) try to infer
            # if it has item_* children, treat as list
            keys = sorted(k for k in g)
            if keys and all(k.startswith("item_") for k in keys):
                return [_read(g[k]) for k in keys]
            return None

        return _read(f)

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
