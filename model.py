from functools import partial

import jax
import jax.numpy as jnp
import utils


def initialize_weights(key, cfg,
                       weights_std, weights_mean=None,
                       layers=('ff', 'rec', 'out')):
    """Initialize weights for the network.

    num_hidden_pre -> num_hidden_post (100 -> 1000) plasticity layer

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
        if std == 'Xavier':  # Use ""Xavier normal"" (paper's Kaiming)
            std = 1 / jnp.sqrt(num_pre + num_post)

        return jax.random.normal(key, shape=(num_pre, num_post)) * std + mean

    ff_key, rec_key, out_key = jax.random.split(key, 3)
    weights = {}

    # Default initial weights mean for training
    if weights_mean is None:
        weights_mean = {'ff': 0, 'rec': 0, 'out': 0}

    # Initialize feedforward weights
    if 'ff' in layers:
        weights['w_ff'] = initialize_layer_weights(ff_key,
                                                   cfg['num_hidden_pre'],
                                                   cfg['num_hidden_post'],
                                                   weights_mean['ff'],
                                                   weights_std['ff'])
        # weights['b_ff'] = jnp.zeros((cfg['num_hidden_post'],))
    if cfg['recurrent'] and 'rec' in layers:
        weights['w_rec'] = initialize_layer_weights(rec_key,
                                                    cfg['num_hidden_post'],
                                                    cfg['num_hidden_post'],
                                                    weights_mean['rec'],
                                                    weights_std['rec'])
        # weights['b_rec'] = jnp.zeros((cfg['num_hidden_post'],))
    if 'out' in layers:
        weights['w_out'] = initialize_layer_weights(out_key,
                                                    cfg['num_hidden_post'],
                                                    1,  # output
                                                    weights_mean['out'],
                                                    weights_std['out'])
        # weights['b_out'] = jnp.zeros((1,))

    return weights

@partial(jax.jit, static_argnames=("cfg"))
def network_forward(key, weights,
                    ff_mask, rec_mask,
                    ff_scale, rec_scale,
                    x, cfg):
    """ Propagate through all layers from input to output. """

    # Add noise to presynaptic layer on each step
    input_noise = jax.random.normal(key, (cfg.num_hidden_pre,))
    x += input_noise * cfg.presynaptic_noise_std

    # Feedforward layer: x -- w_ff --> y

    # Apply scale and sparsity mask to ff weights
    w_ff = weights['w_ff'] * ff_scale * ff_mask

    y = x @ w_ff  # + b

    # Recurrent layer (if present): y -- w_rec --> y

    if cfg.recurrent:
        # Apply scale and sparsity mask to rec weights
        w_rec = weights['w_rec'] * rec_scale * rec_mask
        y += y @ w_rec  # + b

    # Apply nonlinearity
    y = jax.nn.sigmoid(y)

    # Compute output as pre-sigmoid logit (1,) based on postsynaptic layer activity
    output = (y @ weights['w_out']).squeeze()  # + b_out

    return y, output

def compute_decision(key, output):
    """ Make binary decision based on output (probability of decision).
    To lick or not to lick at this step.

    Args:
        key: JAX random key.
        output (1,): Average of postsynaptic activity - probability of decision.

    Returns:
        decision (float): Binary decision (0 or 1).
    """

    return jax.random.bernoulli(key, jax.nn.sigmoid(output)).astype(float)

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

@partial(jax.jit,static_argnames=("plasticity_func", "cfg"))
def update_weights(
    x, y, old_weights,
    reward, expected_reward,
    theta, plasticity_func,
    cfg
):
    """
    Update weights in all layers.

    Calculates full weight matrix regardless of sparsity mask(s).
    Args:
        x (array): x (presynaptic activations)  # TODO rename into x
        y (array): y (postsynaptic activations)
        weights (dict): {w_ff, w_rec, w_out}
        reward (float): Reward at this timestep. TODO Not implemented
        expected_reward (float): Expected reward at this timestep.
        theta (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        cfg (dict): Configuration dictionary.

    Returns: Dictionary of updated weights.
    """
    def update_layer_weights(pre, post, w, r, theta, lr):
            # Allow python 'if' in jitted function because cfg is static
        # Use vectorized volterra_plasticity_function
        if cfg.plasticity_model == "volterra":
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
        elif cfg.plasticity_model == "mlp":
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

    # using expected reward or just the reward:
    # 0 if expected_reward == reward
    reward_term = reward - expected_reward
    # reward_term = reward

    new_weights = {}
    # Update freedforward weights if plastic, else copy old
    if "ff" in cfg.plasticity_layers:
        new_weights['w_ff'] = update_layer_weights(
            x, y, old_weights['w_ff'], reward_term,
            theta, cfg.synapse_learning_rate['ff']  # theta['ff'] TODO
        )
    else:
        new_weights['w_ff'] = old_weights['w_ff']

    # Update recurrent weights if plastic, else copy old or skip if no recurrent
    if "rec" in cfg.plasticity_layers:
        new_weights['w_rec'] = update_layer_weights(
            y, y, old_weights['w_rec'], reward_term,
            theta, cfg.synapse_learning_rate['rec']  # theta['rec'] TODO
        )
    elif cfg.recurrent:
        new_weights['w_rec'] = old_weights['w_rec']

    # Always copy old output weights (never plastic)
    new_weights['w_out'] = old_weights['w_out']

    return new_weights

@partial(jax.jit, static_argnames=("plasticity_func", "cfg", "mode"))
def simulate_trajectory(
    key,
    init_weights,
    ff_mask,
    rec_mask,
    theta,
    plasticity_func,
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
        theta: Plasticity coefficients for the model.
        plasticity_func: Plasticity function to use.
        exp_x (N_sessions, N_steps_per_session_max, N_hidden_pre):
            Presynaptic activity of one exp.
        exp_rewarded_pos (N_sessions, N_steps_per_session_max):
            Rewarded positions of one exp.
        step_mask (N_sessions, N_steps_per_session_max): Valid (not padding) steps
        cfg: Configuration dictionary.
        mode:
            - 'simulation': return trajectories of x, y, output
            - 'generation_train': return x, y, output, decision, rewards
            - 'generation_test': return x, y, output, decision, rewards, weights

    Returns:
        activity_trajec_exp {
            y_trajec_exp (N_sessions, N_steps_per_session_max, N_hidden_post),
            output_trajec_exp (N_sessions, N_steps_per_session_max),
                # If 'generation' in mode:
            decision_trajec_exp (N_sessions, N_steps_per_session_max),
                # If 'test' in mode:
            [weights_trajec_exp {  # Only the plastic layers
                w_ff: (N_sessions, N_steps_per_session_max,
                       N_hidden_pre, N_hidden_post),
                w_rec: (N_sessions, N_steps_per_session_max,
                        N_hidden_post, N_hidden_post)
                }
            ]
        }: Trajectories of activations over the course of the experiment.
    """

    def simulate_session(weights, session_variables):
        """ Simulate trajectory of weights and activations within one session.

        Args:
            weights: Initial weights at the start of the session.
            session_variables: Tuple of (
                session_x,
                session_rewarded_pos,
                session_mask,
                session_keys) for the session.

        Returns:
            weights_session: Parameters at the end of the session.
            activity_trajec_session: {
                y_trajec_session (N_steps_per_session_max, N_hidden_post),
                output_trajec_session (N_steps_per_session_max),
                    # If 'generation' in mode:
                decision_trajec_session (N_steps_per_session_max),
                    # If 'test' in mode:
                weights_trajec_session: {  # Only the plastic layers
                    w_ff: (N_steps_per_session_max, N_hidden_pre, N_hidden_post),
                    w_rec: (N_steps_per_session_max, N_hidden_post, N_hidden_post),
                }
            }"""

        def simulate_step(weights, step_variables):
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
                output_data: {y, output[, decision[, weights]]}
            """
            x, rewarded_pos, valid, step_key = step_variables

            def _do_step(w):
                output_data = {}
                y, output = network_forward(step_key,
                                            w,
                                            ff_mask, rec_mask,
                                            ff_scale, rec_scale,
                                            x, cfg)
                output_data['ys'] = y
                output_data['outputs'] = output

                decision = compute_decision(step_key, output)
                if 'generation' in mode:
                    output_data['decisions'] = decision

                # Reward if lick at rewarded position
                reward = decision * rewarded_pos  # TODO cfg.reward_scale
                expected_reward = compute_expected_reward(reward, None)

                w_updated = update_weights(
                    x, y, w, reward, expected_reward,
                    theta, plasticity_func, cfg)

                if 'test' in mode:
                    output_data['weights'] = {}
                    # Only return trajectories of plastic weights (from updated weights)
                    if "ff" in cfg.plasticity_layers:
                        output_data['weights']['w_ff'] = w_updated['w_ff']
                    if "rec" in cfg.plasticity_layers:
                        output_data['weights']['w_rec'] = w_updated['w_rec']

                return w_updated, output_data

            def _skip_step(w):
                # Nothing is done: return weights unchanged and neutral outputs
                output_data = {}
                # neutral y and output (shapes must match traced shapes)
                output_data['ys'] = jnp.zeros(cfg.num_hidden_post)
                output_data['outputs'] = jnp.array(0.0)

                if 'generation' in mode:
                    output_data['decisions'] = jnp.array(0.0)

                if 'test' in mode:
                    output_data['weights'] = {}
                    # return current (unchanged) weights
                    if "ff" in cfg.plasticity_layers:
                        output_data['weights']['w_ff'] = w['w_ff']
                    if "rec" in cfg.plasticity_layers:
                        output_data['weights']['w_rec'] = w['w_rec']

                return w, output_data

            # Do full computation only when valid is True (real, not padded step)
            return jax.lax.cond(valid, _do_step, _skip_step, weights)

        # Run inner scan over steps within one session
        weights_session, activity_trajec_session = jax.lax.scan(
            simulate_step, weights, session_variables)

        return weights_session, activity_trajec_session

    # Pre-split keys for each session and step
    n_sessions = step_mask.shape[0]
    n_steps = step_mask.shape[1]
    total_keys = int(n_sessions * n_steps)
    flat_keys = jax.random.split(key, total_keys + 1)[1:]
    exp_keys = flat_keys.reshape((n_sessions, n_steps, flat_keys.shape[-1]))

    # Scale ff weights: by constant scale and by number of inputs to each neuron
    n_ff_inputs = ff_mask.sum(axis=0) # N_inputs per postsynaptic neuron
    n_ff_inputs = jnp.where(n_ff_inputs == 0, 1, n_ff_inputs) # avoid /0
    ff_scale = cfg.feedforward_input_scale / jnp.sqrt(n_ff_inputs)[None, :]

    # Scale rec weights: by constant scale and by number of inputs to each neuron
    n_rec_inputs = rec_mask.sum(axis=0) # N_inputs per postsynaptic neuron
    n_rec_inputs = jnp.where(n_rec_inputs == 0, 1, n_rec_inputs) # avoid /0
    rec_scale = cfg.recurrent_input_scale / jnp.sqrt(n_rec_inputs)[None, :]

    # Run outer scan over sessions
    _weights_exp, activity_trajec_exp = jax.lax.scan(
        simulate_session,
        init_weights,
        (exp_x,
         exp_rewarded_pos,
         step_mask,
         exp_keys)
    )

    return activity_trajec_exp
