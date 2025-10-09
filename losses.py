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
    recording_mask,
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
    mask = step_mask[..., None] * recording_mask[None, :]  # (N_sess,N_steps,N_neurons)
    exp_traj_ys_masked = exp_traj_ys * mask
    sim_traj_ys_masked = sim_traj_ys * mask
    se = optax.squared_error(exp_traj_ys_masked, sim_traj_ys_masked)
    mse_loss = jnp.sum(se) / jnp.sum(mask)

    # # TEMP
    # EPS = 1e-12

    # def soft_hist_probs(y, n_bins=32, sigma=None):
    #     """Return soft histogram probs of values in [0,1].
    #     y: any-shape array with values in [0,1]. We flatten.
    #     """
    #     y_flat = jnp.ravel(y)
    #     bins = jnp.linspace(0.0, 1.0, n_bins)
    #     if sigma is None:
    #         sigma = 1.0 / n_bins  # reasonable default
    #     # distances: (n_samples, n_bins)
    #     diffs = y_flat[:, None] - bins[None, :]
    #     weights = jnp.exp(-0.5 * (diffs / sigma) ** 2)
    #     probs = jnp.sum(weights, axis=0)
    #     probs = probs / (jnp.sum(probs) + EPS)
    #     return probs  # shape (n_bins,)

    # def soft_hist_entropy(y, n_bins=32, sigma=None):
    #     p = soft_hist_probs(y, n_bins=n_bins, sigma=sigma)
    #     p = jnp.clip(p, EPS, 1.0)
    #     return -jnp.sum(p * jnp.log(p))  # scalar

    # def edge_penalty(y, eps=0.02):
    #     """Penalty for values within eps of 0 or 1. Smooth squared penalty."""
    #     y = (y - 0.5) ** 2
    #     return jnp.mean(y)
    # y = sim_traj_ys_masked
    # H = soft_hist_entropy(y)
    # edge = edge_penalty(y)
    # H_loss = -H + 3 * edge
    # return H_loss
    return mse_loss

def reinforce_loss(outputs, decisions, rewards, step_mask, lick_cost):
    """ Computes the REINFORCE loss of one experiment trajectory.

    Args:
        outputs (array): (N_sessions, N_steps,) Array of logits (output before sigmoid).
        decisions (array): (N_sessions, N_steps,) Array of binary decisions.
        rewards (array): (N_sessions, N_steps,) Array of rewards at each step.
        step_mask (array): (N_sessions, N_steps,) Mask of valid and padding values.
        lick_cost (float): Cost of each decision.

    Returns:
        loss (float): REINFORCE loss.
        R (float): Total reward.
        D (float): Total number of decisions.
    """
    # Compute log Bernoulli probabilities of the decisions made
    # log(p_t(d_t=1)) = d_t * log(p_t) + (1-d_t) * log(1-p_t) =
    # = d_t * logit - log(1+exp(logit))
    logpi = decisions * outputs - jax.nn.softplus(outputs)
    logpi_sum = jnp.sum(logpi * step_mask)
    R = jnp.sum(rewards * step_mask)
    D = jnp.sum(decisions * step_mask)
    S = R - lick_cost * D
    return -logpi_sum * S, R, D

@partial(jax.jit, static_argnames=["plasticity_funcs", "cfg", "mode"])
def loss(
    key,
    thetas,  # Per-plastic-layer dict of current plasticity coeffs
    weights,  # Current initial weights estimate
    plasticity_funcs,  # Static within losses, per-plastic-layer dict
    exp,
    cfg,  # Static within losses
    mode  # 'training' or 'evaluation'
):
    """
    Computes the total loss for the model on one experiment trajectory.

    Args:
        key (int): Seed for the random number generator.
        thetas (dict): Current plasticity coeffs estimate for each plastic layer,
        weights (dict): Current initial weights estimate for each trainable layer,
            arrays of shape (N_exp, layer_shape...).
        plasticity_funcs (dict): Plasticity functions for each layer.
        exp (dict): Dictionary of variables for one experiment containing:
            'data': dict with keys:
                - 'x_train': Training inputs
                - 'ys': Experimental outputs
                - 'outputs': Model outputs
                - 'decisions': Binary decisions
            'step_mask': Mask to distinguish valid and padding values
            'init_fixed_weights': dict of arrays of fixed initial weights
            'feedforward_mask_training': Feedforward connectivity mask
            'recurrent_mask_training': Recurrent connectivity mask
            'exp_i': Index of the experiment in the batch
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
    init_trainable_weights = {layer: layer_weights[exp['exp_i']]
                              for layer, layer_weights in weights.items()}
    init_weights = {**exp['init_fixed_weights'], **init_trainable_weights}

    reg_theta = 0.0
    thetas_masked = {}
    for layer in thetas:
        # Apply mask to plasticity coefficients to enforce constraints
        if cfg.plasticity_models[layer] == "volterra":
            thetas_masked[layer] = jnp.multiply(thetas[layer],
                                                jnp.array(cfg.coeff_masks[layer]))
        # Compute regularization for theta
        if cfg.reg_types_theta[layer].lower() != "none":
            reg_func = (
                jnp.abs if "l1" in cfg.reg_types_theta[layer].lower()
                else jnp.square
            )
            reg_theta += (cfg.reg_scales_theta[layer]
                         * jnp.sum(reg_func(thetas_masked[layer])))


    # Compute regularization for initial weights and add it to total loss
    reg_w = 0.0
    for layer in init_trainable_weights:
        if cfg.reg_types_weights[layer].lower() != "none":
            reg_func = (
                jnp.abs if "l1" in cfg.reg_types_weights[layer].lower()
                else jnp.square
            )
            reg_w += (cfg.reg_scales_weights[layer]
                      * jnp.sum(reg_func(layer)))

    # Return simulated trajectory of one experiment
    simulated_data = model.simulate_trajectory(
        key,
        init_weights,
        exp['feedforward_mask_training'],
        exp['recurrent_mask_training'],
        thetas_masked,  # Our current plasticity coefficients estimate
        plasticity_funcs,
        exp['data']['x_train'],
        exp['rewarded_pos'],
        exp['step_mask'],
        cfg,
        mode=('simulation' if mode=='training'  # Only return activation trajectories
              else 'generation_test')  # Return both activation and weight trajectories
    )

    # Allow python 'if' in jitted function because cfg is static
    if "neural" in cfg.fit_data:
        neural_loss = neural_mse_loss(
            # subkey,
            exp['data']['ys'],
            simulated_data['ys'],
            exp['step_mask'],
            exp['recording_mask'],
            # cfg.measurement_noise_scale,
        )
    else:
        neural_loss = 0.0

    if "behavioral" in cfg.fit_data:
        behavioral_loss = behavioral_ce_loss(
            simulated_data['outputs'],
            exp['data']['decisions'],
            exp['step_mask']
            )
    else:
        behavioral_loss = 0.0

    if "reinforcement" in cfg.fit_data:
        reinforcement_loss, R, D = reinforce_loss(
            simulated_data['outputs'],
            simulated_data['decisions'],
            simulated_data['rewards'],
            exp['step_mask'],
            cfg.lick_cost
        )
    else:
        reinforcement_loss, R, D = 0.0, 0.0, 0.0

    loss = reg_theta + reg_w + neural_loss + behavioral_loss + reinforcement_loss
    aux = {'trajectories': simulated_data if mode=='evaluation' else None,
           'neural': neural_loss,
           'behavioral': behavioral_loss,
           'total_reward': R,
           'total_licks': D}

    return loss, aux
