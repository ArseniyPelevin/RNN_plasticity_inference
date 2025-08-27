from functools import partial

import jax
import jax.numpy as jnp


def initialize_input_parameters(key, num_inputs, num_pre):
    """Initialize input parameters for the embedding layer.

    num_inputs -> num_hidden_pre (6 -> 100) embedding, fixed for one exp/animal

    Args:
        key: JAX random key.
        num_inputs: Number of input classes.
        num_pre: Number of presynaptic neurons.

    Returns:
        input_params: (num_inputs, num_pre) Array of input parameters.
    """
    input_params = jax.random.normal(key, (num_inputs, num_pre))
    input_params -= jnp.mean(input_params, axis=0, keepdims=True)
    input_params /= jnp.std(input_params, axis=0, keepdims=True) + 1e-8
    return input_params

def initialize_parameters(key, num_pre, num_post, initial_params_scale=0.01):
    """Initialize parameters for the network.

    num_hidden_pre -> num_hidden_post (100 -> 1000) plasticity layer

    Args:
        key: JAX random key.
        num_pre: Number of presynaptic neurons.
        num_post: Number of postsynaptic neurons.
        initial_params_scale: Scale for initializing parameters.

    Returns:
        params: Tuple of (weights, biases) for the layer.
    """
    # TODO loop inside a list for multilayer network
    weights = (
        jax.random.normal(key, shape=(num_pre, num_post))
        * initial_params_scale
    )
    biases = jnp.zeros((num_post,))
    return weights, biases

def embed_inputs_to_presynaptic(key, input_idx, num_input, num_pre, input_params):
    """ Embed input integer into presynaptic layer activity.

    Args:
        key: JAX random key.
        input_idx: Input integer.
        num_input: Number of input classes.
        num_pre: Number of presynaptic neurons.
        input_params: Input parameters for the embedding, fixed for one experiment
            ? and shared with training model ?.

    Returns:
        x: (n_pre,) Array of presynaptic layer activity.
    """
    input_onehot = jax.nn.one_hot(input_idx, num_input).squeeze()
    input_noise = (jax.random.normal(key, (num_pre,)) * 0.1)
    x = jnp.dot(input_onehot, input_params) + input_noise
    return x

def network_forward(x, params):
    """
    Forward pass through the network.
    x -- params --> y

    Args:
        x (num_hidden_pre,): Input array.
        params (tuple): Tuple of (weights, biases). #TODO List of tuples for each layer?

    Returns:
        y (num_hidden_post,): Output array.
    """
    # for w, b in params:
    w, b = params
    return jax.nn.sigmoid(x @ w + b)

def compute_decision(key, y):
    """ Compute binary decision based on output layer activity.
    To lick or not to lick at this step.

    Args:
        key: JAX random key.
        y (num_hidden_post,): Output layer activity.

    Returns:
        decision (float): Binary decision (0 or 1).
    """
    output = jnp.mean(y)
    p_decision = jax.nn.sigmoid(output)
    decision = jax.random.bernoulli(key, p_decision).astype(float)
    return decision

def compute_reward(decision):
    """ Compute reward based on binary decision.

    Args:
        decision (float): Binary decision (0 or 1).

    Returns:
        reward (float): Reward (1 for correct decision, 0 for incorrect).
    """
    # TODO: Implement reward function
    return 0

@partial(jax.jit,static_argnames=("plasticity_func",))
def update_params(
    x, y, params, reward, expected_reward, plasticity_coeffs, plasticity_func, lr=1.0
):
    """
    Updates parameters in one layer
    Inputs:
        #!!! Function redefined from initial to update only one layer!
        x (array): Input for the layer.
        y (array): Output for the layer.
        params (list): Tuple (weights, biases) for one layer.
        reward (float): Reward at this timestep. TODO Not implemented
        expected_reward (float): Expected reward at this timestep.
        plasticity_coeffs (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        lr (float): Learning rate (per input).
    Returns: Updated parameters.
    """

    assert x.ndim == 1, "Input must be a vector"
    assert y.ndim == 1, "Output must be a vector"
    assert len(params) == 2, "Params must be a tuple (weights, biases)"
    assert params[0].shape[0] == x.shape[0], "Input size must match weight shape"
    assert params[0].shape[1] == y.shape[0], "Output size must match weight shape"
    assert params[1].shape[0] == y.shape[0], "Output size must match bias shape"

    lr /= x.shape[0]
    # using expected reward or just the reward:
    # 0 if expected_reward == reward
    reward_term = reward - expected_reward
    # reward_term = reward

    w, b = params
    # vmap over output neurons
    vmap_outputs = jax.vmap(plasticity_func, in_axes=(None, 0, 0, None, None))
    # vmap over input neurons, with nested vmap_outputs it's vmap over each synapse
    vmap_synapses = jax.vmap(vmap_outputs, in_axes=(0, None, 0, None, None))
    dw = vmap_synapses(x, y, w, reward_term, plasticity_coeffs)

    # TODO decide whether to update bias or not
    db = jnp.zeros_like(b)
    # db = vmap_inputs(1.0, reward_term, b, plasticity_coeffs)

    assert (
        dw.shape == w.shape and db.shape == b.shape
    ), "dw and w should be of the same shape to prevent broadcasting \
        while adding"

    params = (
        w + lr * dw,
        b + lr * db,
    )  # TODO rewrite as list comprehension for multilayer

    return params
