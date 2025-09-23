""" Copied originally from https://github.com/yashsmehta/MetaLearnPlasticity.
Modified for our needs"""
from functools import partial

import jax
import jax.numpy as jnp
import model
import optax


def behavioral_ce_loss(logits, decisions, step_mask):
    """
    Functionality: Computes the mean of the element-wise cross entropy
    between decisions and logits.

    Args:
        logits (array): (N_sessions, N_steps) Array of logits (output before sigmoid).
        decisions (array): (N_sessions, N_steps) Array of binary decisions.
        step_mask (array): (N_sessions, N_steps) Mask of valid and padding values.

    Returns: Mean of the element-wise cross entropy.
    """
    losses = optax.sigmoid_binary_cross_entropy(logits, decisions)
    losses = losses * step_mask  # Mask out padding steps

    return jnp.sum(losses) / jnp.sum(step_mask)  # Mean over valid steps only

@jax.jit
def neural_mse_loss(
    # key,
    exp_traj_ys,
    sim_traj_ys,
    step_mask,
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
        step_mask (N_sessions, N_steps):
                Mask to distinguish valid and padding values.
        # recording_sparsity (float): Sparsity of the neural recordings.
        # measurement_noise_scale (float): Scale of the measurement noise.

    Returns: Mean squared error loss for neural activity.
    """
    exp_traj_ys_masked = exp_traj_ys * step_mask[..., None]
    sim_traj_ys_masked = sim_traj_ys * step_mask[..., None]
    se = optax.squared_error(exp_traj_ys_masked, sim_traj_ys_masked)
    norm = jnp.sum(step_mask) * exp_traj_ys.shape[-1]  # N_steps * N_neurons
    mse_loss = jnp.sum(se) / norm
    return mse_loss


@partial(jax.jit, static_argnames=["plasticity_func", "cfg", "mode"])
def loss(
    key,
    init_fixed_weights,
    ff_mask,
    rec_mask,
    theta,  # Current plasticity coeffs, updated on each iteration
    weights,  # Current initial weights estimate, updated on each iteration
    plasticity_func,  # Static within losses
    experimental_data,
    step_mask,
    exp_i,  # Index of the current experiment to extract initial weights from params
    cfg,  # Static within losses
    mode  # 'training' or 'evaluation'
):
    """
    Computes the total loss for the model.

    Args:
        key (int): Seed for the random number generator.
        init_fixed_weights (dict): Dictionary of initial weights for fixed layers.
        ff_mask (array): Feedforward sparsity mask.
        rec_mask (array): Recurrent sparsity mask.
        theta (array): Current plasticity coeffs, updated on each iteration.
        weights (dict): Current initial weights estimate for each trainable layer.
        plasticity_func (function): Plasticity function.
        experimental_data (dict): {"inputs", "xs", "ys", "outputs", "decisions"}.
        step_mask (N_sessions, N_steps): Mask to distinguish valid and padding steps.
        exp_i (int): Index of current experiment to extract init_weights from params.
        cfg (object): Configuration object containing the model settings.
        mode (str): "training" or "evaluation" - decides returning trajectories.

    Returns:
        loss (float): Total loss computed as the sum of theta regularization,
            initial weights regularization, neural loss, and behavioral loss.
        aux (dict): {
            'trajectories': simulated_data if mode=='evaluation' else None,
            'neural': neural_loss if "neural" in cfg.fit_data else 0,
            'behavioral': behavioral_loss if "behavioral" in cfg.fit_data else 0
            }
    """

    # Combine fixed and trainable initial weights for the current experiment
    init_trainable_weights = {layer: layer_weights[exp_i]
                              for layer, layer_weights in weights.items()}
    init_weights = {**init_fixed_weights, **init_trainable_weights}

    # Apply mask to plasticity coefficients to enforce constraints
    if cfg.plasticity_model == "volterra": # Allow 'if' in jitted func: cfg is static
        theta = jnp.multiply(theta, # ['ff'/'rec'] TODO
                             jnp.array(cfg.coeff_mask)) # ['ff'/'rec'] TODO
    # Compute regularization for theta and add it to total loss
    if cfg.regularization_type_theta.lower() != "none":
        reg_func = (
            jnp.abs if "l1" in cfg.regularization_type_theta.lower()
            else jnp.square
        )
        reg_theta = cfg.regularization_scale_theta * jnp.sum(reg_func(theta))
    else:
        reg_theta = 0.0

    # Compute regularization for initial weights and add it to total loss
    if cfg.regularization_type_weights.lower() != "none":
        reg_w = 0.0
        for init_trainable_weights_layer in init_trainable_weights.values():
            reg_func = (
                jnp.abs if "l1" in cfg.regularization_type_weights.lower()
                else jnp.square
            )
            reg_w += (cfg.regularization_scale_weights
                      * jnp.sum(reg_func(init_trainable_weights_layer)))
    else:
        reg_w = 0.0

    # Return simulated trajectory of one experiment
    simulated_data = model.simulate_trajectory(
        key,
        init_weights,
        ff_mask,
        rec_mask,
        theta,  # Our current plasticity coefficients estimate
        plasticity_func,
        experimental_data,
        step_mask,
        cfg,
        mode=('simulation' if mode=='training'  # Only return activation trajectories
              else 'generation_test')  # Return both activation and weight trajectories
    )

    # Allow python 'if' in jitted function because cfg is static
    if "neural" in cfg.fit_data:
        neural_loss = neural_mse_loss(
            # subkey,
            experimental_data['ys'],
            simulated_data['ys'],
            step_mask,
            # cfg.neural_recording_sparsity,
            # cfg.measurement_noise_scale,
        )
    else:
        neural_loss = 0.0

    if "behavioral" in cfg.fit_data:
        behavioral_loss = behavioral_ce_loss(
            simulated_data['outputs'],
            experimental_data['decisions'],
            step_mask
            )
    else:
        behavioral_loss = 0.0

    loss = reg_theta + reg_w + neural_loss + behavioral_loss
    aux = {'trajectories': simulated_data if mode=='evaluation' else None,
           'neural': neural_loss,
           'behavioral': behavioral_loss}

    return loss, aux
