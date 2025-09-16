""" Copied originally from https://github.com/yashsmehta/MetaLearnPlasticity.
Modified for our needs"""
from functools import partial

import jax
import jax.numpy as jnp
import model
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
    input_weights,
    init_fixed_weights,
    ff_mask,
    rec_mask,
    params,  # Current plasticity coeffs, updated on each iteration
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
        input_weights (array): Embedding weights for the inputs.
        init_fixed_weights (dict): Dictionary of initial weights for fixed layers.
        ff_mask (array): Feedforward sparsity mask.
        rec_mask (array): Recurrent sparsity mask.
        params (dict): {"theta": array,
                        "weights": dict{
                            learned_layer: array of shape (N_experiments, ...)
                            }
                        }  # Current parameters being optimized.
        plasticity_func (function): Plasticity function.
        experimental_data (dict): {"inputs", "xs", "ys", "outputs", "decisions"}.
        step_mask (N_sessions, N_steps): Mask to distinguish valid and padding steps.
        exp_i (int): Index of current experiment to extract init_weights from params.
        cfg (object): Configuration object containing the model settings.
        mode (str): "training" or "evaluation" - decides returning trajectories.

    Returns:
        loss (float): Total loss computed as the sum of theta regularization,
            initial weights regularization, neural loss, and behavior loss.
        aux (dict): {'trajectories': simulated_data if mode=='evaluation' else None,
                     'MSE': neural_loss if "neural" in cfg.fit_data else None,
                     'BCE': behavior_loss if "behavior" in cfg.fit_data else None}
              )
    """

    theta = params['theta']
    # Combine fixed and trainable initial weights for the current experiment
    init_trainable_weights = {layer: weights[exp_i]
                              for layer, weights in params['weights'].items()}
    init_fixed_weights = {layer: weights[exp_i]
                          for layer, weights in init_fixed_weights.items()}
    init_weights = {**init_fixed_weights, **init_trainable_weights}

    # Compute regularization for theta and add it to total loss
    if cfg.plasticity_model == "volterra": # Allow 'if' in jitted func: cfg is static
        # Apply mask to plasticity coefficients to enforce constraints
        theta = jnp.multiply(theta, # ['ff'/'rec'] TODO
                             jnp.array(cfg.coeff_mask)) # ['ff'/'rec'] TODO
        if cfg.regularization_type_theta.lower() != "none":
            reg_func = (
                jnp.abs if "l1" in cfg.regularization_type_theta.lower()
                else jnp.square
            )
            loss = cfg.regularization_scale_theta * jnp.sum(reg_func(theta))
        else:
            loss = 0.0

    # Compute regularization for initial weights and add it to total loss
    for init_trainable_weights_layer in init_trainable_weights.values():
        if cfg.regularization_type_weights.lower() != "none":
            reg_func = (
                jnp.abs if "l1" in cfg.regularization_type_weights.lower()
                else jnp.square
            )
            loss += (cfg.regularization_scale_weights
                     * jnp.sum(reg_func(init_trainable_weights_layer)))

    # Return simulated trajectory of one experiment
    simulated_data = model.simulate_trajectory(
        key,
        input_weights,
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
        loss += neural_loss

    if "behavior" in cfg.fit_data:
        behavior_loss = behavior_ce_loss(experimental_data['decisions'],
                                         simulated_data['outputs'])
        loss += behavior_loss
    # loss = regularization + neural_loss + behavior_loss
    
    aux = ({'trajectories': simulated_data if mode=='evaluation' else None,
            'MSE': neural_loss if "neural" in cfg.fit_data else None,
            'BCE': behavior_loss if "behavior" in cfg.fit_data else None}
            )
    
    return loss, aux
