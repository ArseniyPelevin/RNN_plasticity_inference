""" Copied originally from https://github.com/yashsmehta/MetaLearnPlasticity.
Modified for our needs"""
from functools import partial

import jax
import jax.numpy as jnp
import model

# import plasticity.data_loader as data_loader
import optax


def behavior_ce_loss(decisions, logits):
    """
    Functionality: Computes the mean of the element-wise cross entropy
    between decisions and logits.
    Inputs:
        decisions (array): Array of decisions.
        logits (array): Array of logits.
    Returns: Mean of the element-wise cross entropy.
    """
    losses = optax.sigmoid_binary_cross_entropy(logits, decisions)
    return jnp.mean(losses)

@jax.jit
def neural_mse_loss(
    # key,
    exp_traj_ys,
    sim_traj_ys,
    mask,
    # recording_sparsity,
    # measurement_noise_scale,

):
    """
    Functionality: Computes the mean squared error loss for neural activity.
    Args:
        # key (int): Seed for the random number generator.
        exp_traj_ys (N_sessions, N_steps, N_neurons):
                Experimental trajectory of postsynaptic neurons.
        sim_traj_ys (N_sessions, N_steps, N_neurons):
                Simulated trajectory of postsynaptic neurons.
        mask (N_sessions, N_steps):
                Mask to distinguish valid and padding values.
        # recording_sparsity (float): Sparsity of the neural recordings.
        # measurement_noise_scale (float): Scale of the measurement noise.

    Returns: Mean squared error loss for neural activity.
    """
    exp_traj_ys_masked = exp_traj_ys * mask[..., None]
    sim_traj_ys_masked = sim_traj_ys * mask[..., None]
    ce = optax.squared_error(sim_traj_ys_masked, exp_traj_ys_masked)
    mse_loss = jnp.sum(ce) / jnp.sum(mask)
    return mse_loss


@partial(jax.jit, static_argnames=["plasticity_func", "cfg"])
def loss(
    key,
    input_params,
    initial_params,  # TODO update for each epoch/experiment
    plasticity_coeffs,  # Current plasticity coeffs, updated on each iteration
    plasticity_func,  # Static within losses
    inputs,
    exp_traj_ys,  # For computing the neural loss
    exp_traj_decisions,  # For computing the behavior loss
    rewards,
    expected_rewards,
    mask,
    cfg,  # Static within losses
):
    """
    Computes the total loss for the model.

    Args:
        key (int): Seed for the random number generator.
        params (array): Array of parameters.
        plasticity_coeffs (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        xs (array): Array of inputs.
        rewards (array): Array of rewards.
        expected_rewards (array): Array of expected rewards.
        neural_recordings (array): Array of neural recordings.
        decisions (array): Array of decisions.
        cfg (object): Configuration object containing the model settings.

    Returns:
        float: Loss for the cross entropy model.
    """

    if cfg.plasticity_model == "volterra":
        # Apply mask to plasticity coefficients to enforce constraints
        plasticity_coeffs = jnp.multiply(plasticity_coeffs,
                                         jnp.array(cfg.coeff_mask))
        # Compute regularization and add it to total loss
        if cfg.regularization_type.lower() != "none":
            reg_func = (
                jnp.abs if "l1" in cfg.regularization_type.lower()
                else jnp.square
            )
            loss = cfg.regularization_scale * \
                jnp.sum(reg_func(plasticity_coeffs))
        else:
            loss = 0.0

    key, subkey = jax.random.split(key)

    # Return simulated trajectory of one experiment
    _params_final, activations = model.simulate_trajectory(
        subkey,
        input_params,
        initial_params,
        plasticity_coeffs,  # Our current plasticity coefficients estimate
        plasticity_func,
        inputs,  # Data of one whole experiment
        rewards,
        expected_rewards,
        mask,
        cfg
    )

    _sim_traj_xs, sim_traj_ys, sim_traj_outputs = activations

    if "neural" in cfg.fit_data:
        neural_loss = neural_mse_loss(
            # subkey,
            exp_traj_ys,
            sim_traj_ys,
            mask,
            # cfg.neural_recording_sparsity,
            # cfg.measurement_noise_scale,
        )
        loss += neural_loss

    if "behavior" in cfg.fit_data:
        behavior_loss = behavior_ce_loss(exp_traj_decisions, sim_traj_outputs)
        loss += behavior_loss

    # loss = regularization + neural_loss + behavior_loss
    return loss
