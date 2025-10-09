from functools import partial

import jax
import jax.numpy as jnp
import utils


def initialize_weights(key, cfg,
                       weights_std, weights_mean=None,
                       layers=('w_ff', 'w_rec', 'w_out')):
    """Initialize weights for the network.

    Args:
        key: JAX random key.
        cfg: Configuration dictionary.
        weights_std: Dictionary with standard deviations for each layer.
            Either for generation or training.
        weights_mean: (Optional) Dictionary with means for each layer.
            Provided for generation only, defaults to 0 for training.
        layers: Tuple of layer names to initialize. Default is all layers.

    Returns:
        weights (dict): w_ff: (num_hidden_pre, num_hidden_post),
                        w_rec: (num_hidden_post, num_hidden_post) if recurrent,
                        w_out: (num_hidden_post, 1)
                        (b_ff, b_rec, b_out are not used for now)
    """
    def initialize_layer_weights(key, num_pre, num_post, mean, std):
        if std == 'Kaiming':  # Use Kaiming normal
            std = 1 / jnp.sqrt(num_pre)  # If used at all - does not account for mask

        return jax.random.normal(key, shape=(num_pre, num_post)) * std + mean

    ff_key, rec_key, out_key = jax.random.split(key, 3)
    weights = {}

    # Default initial weights mean for training
    if weights_mean is None:
        weights_mean = {'ff': 0, 'rec': 0, 'out': 0}

    # Initialize feedforward weights
    if 'w_ff' in layers:
        weights['w_ff'] = initialize_layer_weights(ff_key,
                                                   cfg['num_hidden_pre'],
                                                   cfg['num_hidden_post'],
                                                   weights_mean['ff'],
                                                   weights_std['ff'])
        # weights['b_ff'] = jnp.zeros((cfg['num_hidden_post'],))
    if cfg['recurrent'] and 'w_rec' in layers:
        weights['w_rec'] = initialize_layer_weights(rec_key,
                                                    cfg['num_hidden_post'],
                                                    cfg['num_hidden_post'],
                                                    weights_mean['rec'],
                                                    weights_std['rec'])
        # weights['b_rec'] = jnp.zeros((cfg['num_hidden_post'],))
    if 'w_out' in layers:
        weights['w_out'] = initialize_layer_weights(out_key,
                                                    cfg['num_hidden_post'],
                                                    1,  # output
                                                    weights_mean['out'],
                                                    weights_std['out'])
        # weights['b_out'] = jnp.zeros((1,))

    return weights

def initialize_trainable_weights(key, cfg, num_experiments, n_restarts=1):
    """ Initialize new initial weights of trainable layers for each experiment.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary.
        num_experiments (int): Number of experiments to initialize weights for.
        n_restarts (int): Number of random trainable weights initializations
            per experiment. Default is 1 for training, can be >1 for evaluation.

    Returns:
        init_trainable_weights (list): per-restart list of per-layer dict of arrays
            of shape (num_experiments, ...)
    """
    # Presplit keys for each restart and experiment
    keys = jax.random.split(key, n_restarts * num_experiments)
    keys = keys.reshape((n_restarts, num_experiments, 2))

    init_trainable_weights = [{layer: [] for layer in cfg.trainable_init_weights}
                              for _ in range(n_restarts)]

    # Different initial weights for each restart of evaluation on test experiments
    for start in range(n_restarts):
        # Different initial weights for each simulated experiment
        for exp in range(num_experiments):
            weights = initialize_weights(keys[start, exp],
                                         cfg,
                                         cfg.init_weights_std_training,
                                         layers=cfg.trainable_init_weights)
            for layer, layer_val in weights.items():
                init_trainable_weights[start][layer].append(layer_val)

        # Convert lists to arrays
        for layer, layer_val in init_trainable_weights[start].items():
            init_trainable_weights[start][layer] = jnp.array(layer_val)

    return init_trainable_weights

@partial(jax.jit, static_argnames=("cfg"))
def network_step(key, weights,
                    ff_mask, rec_mask,
                    ff_scale, rec_scale,
                    x, y_old, cfg):
    """ Propagate through all layers from input to output. """

    # Add noise to presynaptic layer on each step
    input_noise = jax.random.normal(key, (cfg.num_hidden_pre,))
    x += input_noise * cfg.presynaptic_noise_std
    if cfg.input_type == 'task':
        x = jnp.clip(x, min=0)  # Make input positive

    # Feedforward layer: x -- w_ff --> y

    # Apply scale and sparsity mask to ff weights
    w_ff = weights['w_ff'] * ff_scale * ff_mask
    # Compute feedforward activation
    a = x @ w_ff  # + b

    # Recurrent layer (if present): y -- w_rec --> y

    if cfg.recurrent:
        # Apply scale and sparsity mask to rec weights
        w_rec = weights['w_rec'] * rec_scale * rec_mask
        # Add recurrent activation
        a += y_old @ w_rec  # + b

    # Apply nonlinearity
    y = jax.nn.sigmoid(a)

    # Compute output as pre-sigmoid logit (1,) based on postsynaptic layer activity
    output = (y @ weights['w_out']).squeeze()  # + b_out

    return x, y, output

def compute_decision(key, output, min_lick_p):
    """ Make binary decision based on output (probability of decision).
    To lick or not to lick at this step.

    Args:
        key: JAX random key.
        output (1,): Pre-sigmoid logit (float) for decision probability.
        min_lick_p (float): Minimum probability of licking to encourage exploration.
            Used only for reinforcement learning. Set to zero for fitting.

    Returns:
        decision (float): Binary decision (0 or 1).
    """

    return jax.random.bernoulli(key,
                                jnp.maximum(min_lick_p, jax.nn.sigmoid(output)))

def compute_reward(key, decision):
    """ Compute reward based on binary decision.

    Args:
        decision (float): Binary decision (0 or 1).

    Returns:
        reward (float): Reward (1 for correct decision, 0 for incorrect).
    """
    # TODO: Implement reward function
    return jax.random.bernoulli(key).astype(float)

def compute_expected_reward(reward, old_expected_reward):
    """ Compute expected reward based current reward and old expected reward.

    # TODO: Implement expected reward function - Exponential Moving Average?
    """
    return 0

@partial(jax.jit,static_argnames=("plasticity_funcs", "cfg"))
def update_weights(
    x, y, y_old, old_weights, reward_term,
    thetas, plasticity_funcs,
    valid, cfg
):
    """
    Update weights in all layers.

    Calculates full weight matrix regardless of sparsity mask(s).
    Args:
        x (array): x (presynaptic activations)
        y (array): y (postsynaptic activations)
        weights (dict): {w_ff, w_rec, w_out}
        reward_term (float): Reward expectation error, only if licked (d=1).
        theta (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        cfg (dict): Configuration dictionary.

    Returns: Dictionary of updated weights.
    """
    def update_layer_weights(pre, post, w, r,
                             theta, plasticity_func, lr):
            # Allow python 'if' in jitted function because cfg is static
        # Use vectorized volterra_plasticity_function
        if "volterra" in plasticity_func.__name__:
            # Precompute powers of pre, post, w, r
            X = jnp.stack([jnp.ones_like(pre), pre, pre * pre],
                          axis=0)  # (3, N_pre)
            Y = jnp.stack([jnp.ones_like(post), post, post * post],
                          axis=0)  # (3, N_post)
            W = jnp.stack([jnp.ones_like(w), w, w * w],
                          axis=0)  # (3, N_pre, N_post)
            R = jnp.array([1.0, r, r * r])  # (3,)

            dw = plasticity_func(X, Y, W, R, theta)
        # Use per-synapse mlp_plasticity_function
        elif "mlp" in plasticity_func.__name__:
            # vmap over postsynaptic neurons
            vmap_post = jax.vmap(plasticity_func, in_axes=(None, 0, 0, None, None))
            # vmap over presynaptic neurons now, i.e. together vmap over synapses
            vmap_synapses = jax.vmap(vmap_post, in_axes=(0, None, 0, None, None))
            dw = vmap_synapses(pre, post, w, r, theta)

        # TODO decide whether to update bias or not
        # db = jnp.zeros_like(b)
        # db = vmap_post(1.0, reward_term, b, theta)

        assert (
            dw.shape == w.shape  # and db.shape == b.shape
        ), "dw and w should be of the same shape to prevent broadcasting while adding"

        dw *= valid  # Do not update weights on padded steps
        w = utils.softclip(w + lr * dw, cap=cfg.synaptic_weight_threshold)

        return w

    # Will only be checked at compile time
    assert x.ndim == 1, "Input must be a vector"
    assert y.ndim == 1, "Output must be a vector"
    assert old_weights['w_ff'].shape[0] == x.shape[0], \
        "x size must match w_ff shape"
    assert old_weights['w_ff'].shape[1] == y.shape[0], \
        "y size must match w_ff shape"
    if cfg.recurrent:
        assert old_weights['w_rec'].shape[0] == y.shape[0], \
            "y size must match w_rec shape"
        assert old_weights['w_rec'].shape[1] == y.shape[0], \
            "y size must match w_rec shape"

    new_weights = {}
    # Update feedforward weights if plastic, else copy old
    if "ff" in cfg.plasticity_layers:
        ff_layer = 'ff' if 'ff' in thetas else 'both'
        new_weights['w_ff'] = update_layer_weights(
            x, y, old_weights['w_ff'], reward_term,
            thetas[ff_layer],
            plasticity_funcs[0],  # Tuple: 1st element either 'ff' or 'both',
            cfg.synapse_learning_rate
        )
    else:
        new_weights['w_ff'] = old_weights['w_ff']

    # Update recurrent weights if plastic, else copy old or skip if no recurrent
    if "rec" in cfg.plasticity_layers:
        rec_layer = 'rec' if 'rec' in thetas else 'both'
        new_weights['w_rec'] = update_layer_weights(
            y_old, y, old_weights['w_rec'], reward_term,
            thetas[rec_layer],
            (plasticity_funcs[1] if len(plasticity_funcs) > 1
             else plasticity_funcs[0]),  # Tuple: either 2nd element 'rec' or 1st 'both'
            cfg.synapse_learning_rate
        )
    elif cfg.recurrent:
        new_weights['w_rec'] = old_weights['w_rec']

    # Always copy old output weights (never plastic)
    new_weights['w_out'] = old_weights['w_out']

    return new_weights

@partial(jax.jit, static_argnames=("plasticity_funcs", "cfg", "mode"))
def simulate_trajectory(
    key,
    init_weights,
    ff_mask,
    rec_mask,
    thetas,
    plasticity_funcs,
    exp_x,
    exp_rewarded_pos,
    step_mask,
    cfg,
    mode
    ):
    """ Simulate trajectory of activations (and weights) of one experiment (animal).

    Args:
        key: Random key for PRNG.
        init_weights (dict): w_ff: (num_hidden_pre, num_hidden_post),
                w_rec: (num_hidden_post, num_hidden_post) if recurrent,
                w_out: (num_hidden_post, 1)
                (b_ff, b_rec, b_out are not used for now)
        thetas: Per-plastic-layer dict of plasticity coefficients for the model.
        plasticity_funcs: Per-plastic-layer dict of plasticity functions to use.
        exp_x (N_sessions, N_steps_per_session_max, N_hidden_pre):
            Presynaptic activity of one exp.
        exp_rewarded_pos (N_sessions, N_steps_per_session_max):
            Rewarded positions of one exp.
        step_mask (N_sessions, N_steps_per_session_max): Valid (not padding) steps
        cfg: Configuration dictionary.
        mode:
            - 'simulation': return trajectories of y, output
            - 'generation_train': return y, output, decision
            - 'generation_test': return x, y, output, decision, rewards, weights

    Returns:
        activity_trajec_exp {
            y_trajec_exp (N_sessions, N_steps_per_session_max, N_hidden_post),
            output_trajec_exp (N_sessions, N_steps_per_session_max),
                # If 'generation' in mode:
            decision_trajec_exp (N_sessions, N_steps_per_session_max),
                # If 'test' in mode:
            xs_trajec_exp (N_sessions, N_steps_per_session_max, N_hidden_pre),
            rewards_trajec_exp (N_sessions, N_steps_per_session_max),
            weights_trajec_exp {  # Only the plastic layers
                w_ff: (N_sessions, N_steps_per_session_max,
                       N_hidden_pre, N_hidden_post),
                w_rec: (N_sessions, N_steps_per_session_max,
                        N_hidden_post, N_hidden_post)
                }
        }: Trajectories of activations over the course of the experiment.
    """
    y_key, exp_key = jax.random.split(key)
    # Pre-split keys for each session and step
    n_sessions = step_mask.shape[0]
    n_steps = step_mask.shape[1]
    y_keys = jax.random.split(y_key, n_sessions)  # For initial y of each session
    flat_keys = jax.random.split(exp_key, n_sessions * n_steps * 2)  # Two keys per step
    exp_keys = flat_keys.reshape((n_sessions, n_steps, 2, 2))

    # Scale ff weights: by constant scale and by number of inputs to each neuron
    n_ff_inputs = ff_mask.sum(axis=0) # N_inputs per postsynaptic neuron
    n_ff_inputs = jnp.where(n_ff_inputs == 0, 1, n_ff_inputs) # avoid /0
    ff_scale = cfg.feedforward_input_scale / jnp.sqrt(n_ff_inputs)[None, :]

    # Scale rec weights: by constant scale and by number of inputs to each neuron
    n_rec_inputs = rec_mask.sum(axis=0) # N_inputs per postsynaptic neuron
    n_rec_inputs = jnp.where(n_rec_inputs == 0, 1, n_rec_inputs) # avoid /0
    rec_scale = cfg.recurrent_input_scale / jnp.sqrt(n_rec_inputs)[None, :]

    def simulate_session(weights, session_variables):
        """ Simulate trajectory of weights and activations within one session.

        Args:
            weights: Initial weights at the start of the session.
            session_variables: Tuple of (
                session_x,
                session_rewarded_pos,
                session_mask,
                session_keys,
                y_keys) for the session.

        Returns:
            weights_session: Parameters at the end of the session.
            activity_trajec_session: {
                y_trajec_session (N_steps_per_session_max, N_hidden_post),
                output_trajec_session (N_steps_per_session_max),
                    # If 'generation' in mode:
                decision_trajec_session (N_steps_per_session_max),
                    # If 'test' in mode:
                xs_trajec_session (N_steps_per_session_max, N_hidden_pre),
                rewards_trajec_session (N_steps_per_session_max),
                weights_trajec_session: {  # Only the plastic layers
                    w_ff: (N_steps_per_session_max, N_hidden_pre, N_hidden_post),
                    w_rec: (N_steps_per_session_max, N_hidden_post, N_hidden_post),
                }
            }"""

        def simulate_step(step_carry, step_variables):
            """ Simulate forward pass within one time step.

            Args:
                weights (N_hidden_pre, N_hidden_post): carry for jax.lax.scan.
                step_variables: (
                    x,
                    rewarded_pos,
                    valid,
                    step_key)

            Returns:
                weights: Updated weighteters after the step.
                output_data: {y, - (N_hidden_post,) in (0, 1)
                              output[, - (1,) pre-sigmoid logit (float)
                                # If 'generation' in mode:
                              decision[, - (1,) boolean decision
                                # If 'test' in mode:
                              xs, - (N_hidden_pre,) for debugging
                              rewards, - (1,) boolean reward
                              weights]] - (dict of plastic layers)
                              }
            """
            w, y_old = step_carry
            x_input, rewarded_pos, valid, keys = step_variables

            output_data = {}
            x, y_new, output = network_step(keys[0],
                                            w,
                                            ff_mask, rec_mask,
                                            ff_scale, rec_scale,
                                            x_input,  # Clean input without noise
                                            y_old,
                                            cfg)
            output_data['ys'] = y_new
            output_data['outputs'] = output

            decision = compute_decision(keys[1], output, cfg.min_lick_probability)
            if ('generation' in mode  # For generation/evaluation/debugging
                or "reinforcement" in cfg.fit_data  # For reinforcement learning
                ):
                output_data['decisions'] = decision

            # Reward if lick at rewarded position
            reward = decision * rewarded_pos  # TODO cfg.reward_scale
            # expected_reward = compute_expected_reward(reward, 0)  # TODO?
            # Treat lick probability as expected reward
            expected_reward = jax.nn.sigmoid(output)
            # Reward expectation error, only if licked
            reward_term = (reward - expected_reward) * decision

            w_updated = update_weights(
                x, y_new, y_old, w, reward_term,
                thetas, plasticity_funcs, valid, cfg)

            if 'test' in mode:
                output_data['xs'] = x  # For debugging
                output_data['weights'] = {}  # For evaluation/debugging
                # Only return trajectories of plastic weights (from updated weights)
                if "ff" in cfg.plasticity_layers:
                    output_data['weights']['w_ff'] = w_updated['w_ff']
                if "rec" in cfg.plasticity_layers:
                    output_data['weights']['w_rec'] = w_updated['w_rec']

            if ('test' in mode  # For evaluation/debugging
                or 'reinforcement' in cfg.fit_data  # For reinforcement learning
                ):
                output_data['rewards'] = reward

            return (w_updated, y_new), output_data

        *session_variables, y_key = session_variables
        # Initialize y activity at start of session
        init_y = jax.random.normal(y_key, (cfg.num_hidden_post,))
        init_y = jax.nn.sigmoid(init_y)  # Initial activity between 0 and 1

        # Run inner scan over steps within one session
        (weights, _), session_output = jax.lax.scan(
            simulate_step, (weights, init_y), session_variables)

        return weights, session_output

    # Run outer scan over sessions within one experiment
    _carry, activity_trajec_exp = jax.lax.scan(
        simulate_session,
        init_weights,
        (exp_x,
         exp_rewarded_pos,
         step_mask,
         exp_keys,
         y_keys)
    )

    # Zero out padding steps in trajectories
    for name, traj in activity_trajec_exp.items():
        if name != 'weights':
            activity_trajec_exp[name] = (traj * step_mask[..., None]
                                         if name in ['xs', 'ys'] else traj * step_mask)

    return activity_trajec_exp
