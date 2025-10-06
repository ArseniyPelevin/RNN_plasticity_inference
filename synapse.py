"""Module handling plasticity.
Copied originally from https://github.com/yashsmehta/MetaLearnPlasticity.
Modified for our needs"""

import re

import jax
import jax.numpy as jnp
import numpy as np
from utils import generate_gaussian, standardize_coeff_init


def volterra_plasticity_function(X, Y, W, R, coeff):
    """
    Vectorized Volterra/Taylor plasticity:
      dw_{j,i} = sum_{a,b,c,d=0..2} coeff[a,b,c,d] * x_j^a * y_i^b * w_{j,i}^c * r^d
    Shapes:
      X: (3, N_pre,), Y: (3, N_post,), W: (3, N_pre, N_post), R: (3,), coeff: (3,3,3,3)
      returns dw: (N_pre, N_post)
    """
    # successive contractions to keep memory stable:
    # 1) contract coeff over r-powers -> (3,3,3)
    theta_R = jnp.tensordot(coeff, R, axes=(3, 0))  # (a,b,c)
    # 2) mix in weight powers -> (3,3,N_pre,N_post)
    theta_WR = jnp.einsum('abc,cji->abji', theta_R, W)  # (a,b,j,i)
    # 3) combine with x- and y-powers -> (N_pre, N_post)
    dw = jnp.einsum('aj,bi,abji->ji', X, Y, theta_WR)  # (j,i)

    # Use if NaN needs to be inspected
    # _ = jax.lax.cond(jnp.any(jnp.isnan(dw)),
    #                  lambda: (jax.debug.print('In plasticity_function dw is NaN:\n',
    #     'X:\n{}\nY:\n{}\nW:\n{}\nR:\n{}\ncoeff:\n{}\ntheta_R:\n{}\ntheta_WR:\n{}\n\n',
    #     X, Y, W, R, coeff, theta_R, theta_WR), 0)[1],
    #                  lambda: 0)
    return dw

def volterra_synapse_tensor(x, y, w, r):
    """
    Computes the Volterra synapse tensor for given inputs.
    Args:
        x, y, w, r (floats): Inputs to the Volterra synapse tensor.
    Returns: A 3x3x3x3 tensor representing the Volterra synapse tensor.
    """
    synapse_tensor = jnp.array(
        [
            [
                [[x**i * y**j * w**k * r**l for i in range(3)] for j in range(3)]
                for k in range(3)
            ]
            for l in range(3)
        ]
    )
    return synapse_tensor

def volterra_plasticity_function_old(x, y, w, r, volterra_coefficients):
    """
    Computes the Volterra plasticity function for given inputs and coefficients.
    Args:
        x, y, w, r (floats): Inputs to the Volterra plasticity function.
        volterra_coefficients (array): Coefficients for Volterra plasticity function.
    Returns: The result of the Volterra plasticity function.
    """
    synapse_tensor = volterra_synapse_tensor(x, y, w, r)
    dw = jnp.sum(jnp.multiply(volterra_coefficients, synapse_tensor))
    return dw


def mlp_forward_pass(mlp_params, inputs):
    """
    Performs a forward pass through a multi-layer perceptron (MLP).
    Args:
        mlp_params (list): List of tuples (weights, biases) for each layer.
        inputs (array): Input data.
    Returns: The logits output of the MLP.
    """
    activation = inputs
    for w, b in mlp_params[:-1]:  # for all but the last layer
        activation = jax.nn.leaky_relu(jnp.dot(activation, w) + b)
    final_w, final_b = mlp_params[-1]  # for the last layer
    logits = jnp.dot(activation, final_w) + final_b
    output = jnp.tanh(logits)
    return jnp.squeeze(output)


def mlp_plasticity_function(x, y, w, r, mlp_params):
    """
    Computes the MLP plasticity function for given inputs and MLP parameters.
    Args:
        x, y, w, r (floats): Inputs to the MLP plasticity function.
        mlp_params (list): MLP parameters.
    Returns: The result of the MLP plasticity function.
    """
    inputs = jnp.array([x, y, w, r])
    dw = mlp_forward_pass(mlp_params, inputs)
    return dw


def init_zeros():
    return np.zeros((3, 3, 3, 3))


def init_random(key, scale):
    return generate_gaussian(key, (3, 3, 3, 3), scale=scale)


def split_init_string(s):
    """
    Splits an initialization string into a list of matches.
    Args:
        s (str): Initialization string.
    Returns: A list of matches.
    """
    return [
        match.replace(" ", "")
        for match in re.findall(r"(-?\s*[A-Za-z0-9.]+[A-Za-z][0-9]*)", s)
    ]


def extract_numbers(s):
    """
    Extracts numbers from string initialization: X1R0W1
    for the plasticity coefficients
    Args:
        s (str): String to extract numbers from.
    Returns: A tuple of extracted numbers.
    """
    x = int(re.search(r"X(\d+)", s).group(1))
    y = int(re.search(r"Y(\d+)", s).group(1))
    w = int(re.search(r"W(\d+)", s).group(1))
    r = int(re.search(r"R(\d+)", s).group(1))
    multiplier_match = re.search(r"^(-?\d+\.?\d*)", s)
    multiplier = float(multiplier_match.group(1)) if multiplier_match else 1.0
    assert x < 3 and y < 3 and w < 3 and r < 3, "X, Y, W, R must be between 0 and 2"
    return x, y, w, r, multiplier


def init_generation_volterra(init):
    """
    Initializes the parameters for the Volterra generation model.
    Args:
        init (str): Initialization string.
    Returns:
        parameters (ndarray): Initialized Volterra parameters
        volterra_plasticity_function
    """
    parameters = np.zeros((3, 3, 3, 3))
    inits = split_init_string(init)
    for init in inits:
        x, y, w, r, multiplier = extract_numbers(init)
        parameters[x][y][w][r] = multiplier

    return jnp.array(parameters), volterra_plasticity_function


def init_plasticity_volterra(key, init, scale):
    """
    Initializes the parameters for the Volterra plasticity model.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        init (str): Initialization method, either "zeros" or "random".

    Returns:
        parameters (ndarray): Initialized Volterra parameters
        volterra_plasticity_function
    """
    init_functions = {
        "zeros": init_zeros,
        "random": lambda: init_random(key, scale=scale),
    }

    parameters = init_functions[init]()
    return jnp.array(parameters), volterra_plasticity_function


def init_plasticity_mlp(key, layer_sizes, scale=0.01):
    """
    Initializes the parameters for a multi-layer perceptron (MLP) plasticity model.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        layer_sizes (list of int): sizes of each layer in the MLP.
        scale (float): Scale of Gaussian to initialize parameters. Default 0.01.

    Returns:
        mlp_params: A list of tuples of weights and biases for each layer in the MLP.
        mlp_plasticity_function
    """
    w_key, b_key = jax.random.split(key)
    mlp_params = [
        (
            generate_gaussian(w_key, (m, n), scale),
            generate_gaussian(b_key, (n,), scale),
        )
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:], strict=False)
    ]
    return mlp_params, mlp_plasticity_function


def init_plasticity(key, cfg, mode):
    """
    Initializes the parameters for a given plasticity model.
    Args:
        key (int): Seed for the random number generator.
        cfg (object): Configuration object containing the model settings.
        mode (str): Mode of operation ("generation" or "plasticity").

    Returns:
        generation_thetas: Per-plastic-layer dict of initialized parameters.
        plasticity_funcs: Per-plastic-layer dict of plasticity functions.
    """
    generation_thetas, plasticity_funcs = {}, {}
    if "generation" in mode:
        # Only for plastic layers
        for layer in cfg.plasticity_layers:
            key, subkey = jax.random.split(key)
            if cfg.generation_model[layer] == "volterra":
                # Ensure the form "X1Y0W0R1"
                cfg.generation_plasticity[layer] = standardize_coeff_init(
                    cfg.generation_plasticity[layer])
                (generation_thetas[layer],
                 plasticity_funcs[layer]) = init_generation_volterra(
                     init=cfg.generation_plasticity[layer])
            elif cfg.generation_model[layer] == "mlp":
                raise NotImplementedError("MLP generation model not implemented")
                (generation_thetas[layer],
                 plasticity_funcs[layer]) = init_plasticity_mlp(
                     subkey, cfg.meta_mlp_layer_sizes)
    elif "plasticity" in mode:
        for layer in cfg.plasticity_layers:
            key, subkey = jax.random.split(key)
            if cfg.plasticity_model[layer] == "volterra":
                (generation_thetas[layer],
                 plasticity_funcs[layer]) = init_plasticity_volterra(
                     subkey,
                     init=cfg.plasticity_coeffs_init,
                     scale=cfg.plasticity_coeffs_init_scale[layer])
            elif cfg.plasticity_model[layer] == "mlp":
                raise NotImplementedError("MLP plasticity model not implemented")
                (generation_thetas[layer],
                 plasticity_funcs[layer]) = init_plasticity_mlp(
                     subkey, cfg.meta_mlp_layer_sizes)

    if generation_thetas:
        return generation_thetas, plasticity_funcs
    else:
        raise RuntimeError(
            "mode needs to be either generation or plasticity, "
            "and plasticity_model needs to be either volterra or mlp"
        )
