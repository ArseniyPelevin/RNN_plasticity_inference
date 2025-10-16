import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import simulation


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
        recording_mask (N_neurons): Sparsity of the neural recordings.
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

@eqx.filter_jit
def loss(
    params,
    key,
    exp,
    plasticity,
    cfg,
    returns
):
    """
    Computes the total loss for the model on one experiment trajectory.

    Args:
        params (dict): Current estimates of the parameters to be optimized:
            'thetas': dict of plasticity coefficients for each plastic layer,
            'w_init_learned': dict of initial weights for each trainable layer.
        key (int): Seed for the random number generator.
        exp: Experiment object containing the data and network.
        plasticity: dict of plasticity modules for each plastic layer.
        cfg: Configuration object.
        returns: Tuple of strings indicating which outputs to return.

    Returns:
        loss (float): Total loss computed as the sum of theta regularization,
            initial weights regularization, neural loss, and behavioral loss.
        aux (dict): {
            'trajectories': simulated_data if mode=='evaluation' else None,
            'neural': neural_loss if "neural" in cfg.fit_data else 0,
            'behavioral': behavioral_loss if "behavioral" in cfg.fit_data else 0
            }
    """
    thetas = params['thetas']
    w_init_learned = params['w_init_learned'][exp.exp_i]

    # Update plasticity coefficients with current estimates.
    for layer in thetas:
        # Apply mask to plasticity coefficients to enforce constraints
        if cfg.plasticity.plasticity_models[layer] == "volterra":
            thetas[layer] *= plasticity[layer].coeff_mask
        plasticity[layer] = eqx.tree_at(
            lambda p: p.coeffs, plasticity[layer], thetas[layer])
    # Update initial weights of trainable layers with current estimates.
    w_init = {**exp.w_init_train, **w_init_learned}  # Second overrides first
    # Apply updated weights to experiment's network
    exp = eqx.tree_at(lambda exp: exp.network, exp, exp.network.apply_weights(w_init))

    reg_theta = 0.0
    # Compute regularization for theta
    if cfg.training.reg_types_theta.lower() != "none":
        reg_func = (
            jnp.abs if "l1" in cfg.training.reg_types_theta.lower()
            else jnp.square
        )
        reg_theta += jnp.sum(  # Sum over all leaves of thetas
            jnp.sum(reg_func(x)) for x in jax.tree_util.tree_leaves(thetas)
            ) * cfg.training.reg_scales_theta

    # Compute regularization for initial weights and add it to total loss
    reg_w = 0.0
    if cfg.training.reg_types_weights.lower() != "none":
        reg_func = (
            jnp.abs if "l1" in cfg.training.reg_types_weights.lower()
            else jnp.square
        )
        reg_w += jnp.sum(  # Sum over all leaves of w_init_learned
            jnp.sum(reg_func(x)) for x in jax.tree_util.tree_leaves(w_init_learned)
            ) * cfg.training.reg_scales_weights

    # Return trajectories that caller wants, and ones we need for loss computation
    simulation_returns = returns
    if 'neural' in cfg.training.fit_data:
        simulation_returns = simulation_returns + ('ys',)
    if 'behavioral' in cfg.training.fit_data:
        simulation_returns = simulation_returns + ('outputs',)
    if 'reinforcement' in cfg.training.fit_data:
        simulation_returns = simulation_returns + ('outputs', 'decisions', 'rewards')

    # Return simulated trajectory of one experiment
    simulated_data = simulation.simulate_trajectory(
        key,
        exp,
        exp.x_input,
        exp.network,
        plasticity,
        returns=simulation_returns
    )

    # Allow python 'if' in jitted function because cfg is static
    if "neural" in cfg.training.fit_data:
        neural_loss = neural_mse_loss(
            # subkey,
            exp.data['ys'],
            simulated_data['ys'],
            exp.step_mask,
            exp.network.recording_mask,
            # cfg.measurement_noise_scale,
        )
    else:
        neural_loss = 0.0

    if "behavioral" in cfg.training.fit_data:
        behavioral_loss = behavioral_ce_loss(
            simulated_data['outputs'],
            exp.data['decisions'],
            exp.step_mask
            )
    else:
        behavioral_loss = 0.0

    if "reinforcement" in cfg.training.fit_data:
        reinforcement_loss, R, D = reinforce_loss(
            simulated_data['outputs'],
            simulated_data['decisions'],
            simulated_data['rewards'],
            exp.step_mask,
            cfg.training.lick_cost
        )
    else:
        reinforcement_loss, R, D = 0.0, 0.0, 0.0

    # Compute total loss
    loss = reg_theta + reg_w + neural_loss + behavioral_loss + reinforcement_loss

    # Only return trajectories that caller wants
    simulated_data = {k: v for k, v in simulated_data.items() if k in returns}

    aux = {'trajectories': simulated_data,
           'neural': neural_loss,
           'behavioral': behavioral_loss,
           'total_reward': R,
           'total_licks': D}

    return loss, aux
