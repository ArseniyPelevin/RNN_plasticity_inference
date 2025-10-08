"""Module with utility functions, taken as is from https://github.com/yashsmehta/MetaLearnPlasticity"""

import ast
import inspect
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any
import h5py

import jax
import jax.numpy as jnp
import numpy as np
from colorama import Back, Fore, Style, init
from scipy.special import kl_div

# Initialize colorama
init(autoreset=True)

# Define color codes for logging using colorama
COLOR_CODES = {
    logging.DEBUG: Fore.WHITE,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Back.RED + Fore.WHITE,
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        print(
            f"{inspect.stack()[1].frame.f_globals.get('__name__','<module>').rsplit('.',1)[-1]}.{inspect.stack()[1].function} -> {inspect.stack()[0].frame.f_globals.get('__name__','<module>').rsplit('.',1)[-1]}.{inspect.stack()[0].function}"
        )
        color = COLOR_CODES.get(record.levelno, "")
        # Format the message and levelname with color
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def setup_logging(level=logging.INFO) -> None:
    """Set up logging with colored output."""
    
    handler = logging.StreamHandler()
    formatter = ColoredFormatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []  # Remove any existing handlers
    root.addHandler(handler)


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


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def compute_neg_log_likelihoods(ys, decisions):
    """
    Computes the negative log-likelihoods for a set of predictions and decisions.

    Args:
        ys (array-like): Array of predicted probabilities.
        decisions (array-like): Array of binary decisions (0 or 1).

    Returns:
        float: The mean negative log-likelihood.
    """
    
    not_ys = jnp.ones_like(ys) - ys
    neg_log_likelihoods = -2 * jnp.log(jnp.where(decisions == 1, ys, not_ys))
    return jnp.mean(neg_log_likelihoods)


def kl_divergence(logits1, logits2):
    """
    Computes the Kullback-Leibler (KL) divergence between two sets of logits.

    Args:
        logits1 (array-like): The first set of logits.
        logits2 (array-like): The second set of logits.

    Returns:
        float: The sum of the KL divergence between the two sets of logits.
    """
    
    p = jax.nn.softmax(logits1)
    q = jax.nn.softmax(logits2)
    kl_matrix = kl_div(p, q)
    return np.sum(kl_matrix)
    
def softclip(x, cap, p=10):
    return x / ((1.0 + jnp.abs(x / cap) ** p) ** (1.0 / p))

def create_nested_list(num_outer, num_inner):
    """
    Creates a nested list with specified dimensions.

    Args:
        num_outer (int): The number of outer lists.
        num_inner (int): The number of inner lists within each outer list.

    Returns:
        list: A nested list where each outer list contains `num_inner` empty lists.
    """
    
    return [[[] for _ in range(num_inner)] for _ in range(num_outer)]


def truncated_sigmoid(x, epsilon=1e-6):
    """
    Applies a sigmoid function to the input and truncates the output to a specified range.

    Args:
        x (array-like): Input array.
        epsilon (float, optional): Small value to ensure the output is within the range (epsilon, 1 - epsilon). Default is 1e-6.

    Returns:
        array-like: The truncated sigmoid output.
    """
    print(
        f"{inspect.stack()[1].frame.f_globals.get('__name__','<module>').rsplit('.',1)[-1]}.{inspect.stack()[1].function} -> {inspect.stack()[0].frame.f_globals.get('__name__','<module>').rsplit('.',1)[-1]}.{inspect.stack()[0].frame.f_globals.get('__name__','<module>').rsplit('.',1)[-1]}.{inspect.stack()[0].function}"
    )
    return jnp.clip(jax.nn.sigmoid(x), epsilon, 1 - epsilon)

def sample_truncated_normal(key, mean, std, shape=1):
    """ Samples values from a normal distribution that are >= (mean - std). """
    samples = jax.random.normal(key, shape)
    while not jnp.all(samples >= -1):
        key, subkey = jax.random.split(key)
        samples2 = jax.random.normal(subkey, shape)
        samples = jnp.where(samples < -1, samples2, samples)
    samples = samples * std + mean
    return samples.round().astype(jnp.int32)

def experiment_lists_to_tensors(nested_lists):
    """
    Converts a tuple of nested lists of one experiment's data into a tuple of tensors.
    Pads shorter trials with zeros.

    Args:
        nested_list (list): Tuple of nested lists of all variables.

    Returns:
        tensors (tuple): A tuple of tensor (jnp.ndarray) representation
            of the nested list, padded with zeros.
        mask (jnp.ndarray): A mask indicating valid (not padded) time steps.
        steps_per_session (jnp.ndarray): The number of steps per session.
    """

    def experiment_list_to_tensor(nested_list):
        """ Converts a nested list into a per-session tensor for one variable. """

        # infer variable shape from first element
        element_shape = np.asarray(nested_list[0][0][0]).shape
        tensor = np.zeros((num_sessions, max_steps_per_session, *element_shape), dtype=float)

        for i in range(num_sessions):
            offset = 0
            for j in range(len(nested_list[i])):
                trial = nested_list[i][j]
                for k in range(len(trial)):
                    tensor[i, offset + k, ...] = np.asarray(trial[k], dtype=float)
                offset += len(trial)

        return jnp.array(tensor)

    inputs, xs, ys, decisions, rewards, expected_rewards = nested_lists

    # Use inputs as proxy for session/trial structure
    num_sessions = len(inputs)
    # total number of steps per session (concatenating trials within session)
    steps_per_session = [sum(len(trial) for trial in inputs[s]) for s in range(num_sessions)]
    steps_per_session = jnp.array(steps_per_session)
    max_steps_per_session = jnp.max(steps_per_session)

    mask = (jnp.arange(max_steps_per_session)[None, :] < steps_per_session[:, None])

    # max_trial_length = int(
    #         max(len(inputs[s][t]) for s in range(num_sessions) 
    #             for t in range(len(inputs[s])))
    #     )
    tensors = [
        experiment_list_to_tensor(var) for var in
        [inputs, xs, ys, decisions, rewards, expected_rewards]
    ]

    return tensors, mask, steps_per_session

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
        if cfg.plasticity_models[layer] == "volterra":
            coeff_mask = jnp.array(cfg.coeff_masks[layer])
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
        csv_file = logdata_path / f"exp_{cfg.expid}.csv"
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

def save_nested_hdf5(obj, fname):
    """Save a nested dict of arrays to HDF5, preserving the tree."""
    with h5py.File(fname, "w") as f:
        def _recursively_write(group, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    subgroup = group.create_group(k)
                    _recursively_write(subgroup, v)
                else:
                    arr = np.array(jax.device_get(v))
                    group.create_dataset(k, data=arr, compression="gzip", compression_opts=4)
        _recursively_write(f, obj)

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

def standardize_coeff_init(coeff_init):
    """
    Standardizes the coefficient initialization string by ensuring each variable (X, Y, W, R) is followed by its power.

    Args:
        coeff_init (str): The coefficient initialization string to be standardized.

    Returns:
        str: The standardized coefficient initialization string.
    """
    
    terms = re.split(r"(?=[+-])", coeff_init)
    formatted_terms = []
    for term in terms:
        var_dict = {"X": 0, "Y": 0, "W": 0, "R": 0}
        number_prefix = re.match(r"[+-]?\d*\.?\d*", term).group(0)
        parts = re.findall(r"([+-]?\d*\.?\d*)([XYWR])(\d*)", term)
        for _, var, power in parts:
            power = int(power) if power else 1
            var_dict[var] = power
        formatted_term = number_prefix + "".join(
            [f"{key}{val}" for key, val in var_dict.items()]
        )
        formatted_terms.append(formatted_term)

    standardized_coeff_init = "".join(formatted_terms)
    return standardized_coeff_init
