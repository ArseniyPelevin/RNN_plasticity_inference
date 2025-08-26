import jax
import jax.numpy as jnp


def update_params(
    x, y, params, reward, expected_reward, plasticity_coeffs, plasticity_func, lr=1.0
):
    """
    Functionality: Updates the parameters in one layer
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
    vmap_inputs = jax.vmap(plasticity_func, in_axes=(None, 0, 0, None, None))
    # vmap over input neurons
    vmap_synapses = jax.vmap(vmap_inputs, in_axes=(0, None, 0, None, None))
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
