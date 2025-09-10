from functools import partial

import jax
import jax.numpy as jnp
import utils


def initialize_input_parameters(key, num_inputs, num_pre, input_params_scale):
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
    # Standardize for each neuron across all classes
    input_params -= jnp.mean(input_params, axis=0, keepdims=True)
    input_params /= jnp.std(input_params, axis=0, keepdims=True) + 1e-8
    return input_params * input_params_scale

def initialize_parameters(key, cfg):
    """Initialize parameters for the network.

    num_hidden_pre -> num_hidden_post (100 -> 1000) plasticity layer

    Args:
        key: JAX random key.
        cfg: Configuration dictionary.

    Returns:
        params (dict): w_ff: (num_hidden_pre, num_hidden_post),
                       w_rec: (num_hidden_post, num_hidden_post) if recurrent,
                       w_out: (num_hidden_post, 1)
                       (b_ff, b_rec, b_out are not used for now)
    """
    def initialize_layer_params(key, num_pre, num_post, scale):
        if scale == 'Xavier':  # Use ""Xavier normal"" (paper's Kaiming)
            scale = 1 / jnp.sqrt(num_pre + num_post)

        return jax.random.normal(key, shape=(num_pre, num_post)) * scale

    ff_key, rec_key, out_key = jax.random.split(key, 3)
    params = {}

    # Initialize feedforward weights
    params['w_ff'] = initialize_layer_params(ff_key,
                                             cfg['num_hidden_pre'],
                                             cfg['num_hidden_post'],
                                             cfg['init_params_scale']['ff'])
    # params['b_ff'] = jnp.zeros((cfg['num_hidden_post'],))
    if cfg['recurrent']:  # Whether there is recurrent connectivity at all
        params['w_rec'] = initialize_layer_params(rec_key,
                                                  cfg['num_hidden_post'],
                                                  cfg['num_hidden_post'],
                                                  cfg['init_params_scale']['rec'])
        # params['b_rec'] = jnp.zeros((cfg['num_hidden_post'],))
    else:
        params['w_rec'] = None
        # params['b_rec'] = None
    params['w_out'] = initialize_layer_params(out_key,
                                              cfg['num_hidden_post'],
                                              1,
                                              cfg['init_params_scale']['out'])
    # params['b_out'] = jnp.zeros((1,))

    return params

@partial(jax.jit, static_argnames=("cfg"))
def network_forward(key, input_params, params, ff_mask, rec_mask, step_input, cfg):
    """ Propagate through all layers from input to output. """

    # # Embed input integer into presynaptic layer activity
    # input_onehot = jax.nn.one_hot(step_input, cfg.num_inputs).squeeze()
    # x = jnp.dot(input_onehot, input_params)

    # Makeshift for input firing (TODO)  # x IS input firing
    x = step_input

    # Add noise to presynaptic layer on each step
    input_noise = jax.random.normal(key, (cfg.num_hidden_pre,))
    x += input_noise * cfg.presynaptic_noise_std

    # Feedforward layer: x -- w_ff --> y

    # Scale ff weights: by constant scale (and by number of inputs?)
    # n_ff_inputs = ff_mask.sum(axis=0) # N_inputs per postsynaptic neuron
    # n_ff_inputs = jnp.where(n_ff_inputs == 0, 1, n_ff_inputs) # avoid /0
    w_ff = params['w_ff'] * cfg.feedforward_input_scale  # / n_ff_inputs
    w_ff *= ff_mask  # Apply feedforward sparsity mask
    y = x @ w_ff  # + b

    # Recurrent layer (if present): y -- w_rec --> y

    if cfg.recurrent:
        # Scale rec weights: by const scale (and by num of inputs?)
        # n_rec_inputs = rec_mask.sum(axis=0) # N_inputs per postsynaptic neuron
        # n_rec_inputs = jnp.where(n_rec_inputs == 0, 1, n_rec_inputs) # avoid /0
        w_rec = params['w_rec'] * cfg.recurrent_input_scale  # / n_rec_inputs
        w_rec *= rec_mask
        y += y @ w_rec  # + b

    # Apply nonlinearity
    y = jax.nn.sigmoid(y)

    # Compute output probability ((1,) logit) based on postsynaptic layer activity
    output = jax.nn.sigmoid(y @ params['w_out']).squeeze()  # + b_out

    return x, y, output

def compute_decision(key, output):
    """ Make binary decision based on output (probability of decision).
    To lick or not to lick at this step.

    Args:
        key: JAX random key.
        output (1,): Average of postsynaptic activity - probability of decision.

    Returns:
        decision (float): Binary decision (0 or 1).
    """

    return jax.random.bernoulli(key, output).astype(float)

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
def update_params(
    x, y, params,
    reward, expected_reward,
    theta, plasticity_func,
    cfg
):
    """
    Update parameters in all layers.

    Calculates full weight matrix regardless of sparsity mask(s).
    Args:
        #!!! Function redefined from initial to update only one layer!
        x (array): x (presynaptic activations)  # TODO rename into x
        y (array): y (postsynaptic activations)
        params (dict): {w_ff, w_rec, w_out}
        reward (float): Reward at this timestep. TODO Not implemented
        expected_reward (float): Expected reward at this timestep.
        theta (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        cfg (dict): Configuration dictionary.

    Returns: Dictionary of updated parameters.
    """
    def update_layer_params(pre, post, w, r, theta, lr):
            # Allow python 'if' in jitted function because cfg is static
        # Use vectorized volterra_plasticity_function
        if cfg.plasticity_model == "volterra":
            dw = plasticity_func(pre, post, w, reward_term, theta)
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
    assert params['w_ff'].shape[0] == x.shape[0], "x size must match w_ff shape"
    assert params['w_ff'].shape[1] == y.shape[0], "y size must match w_ff shape"
    if cfg.recurrent:
        assert params['w_rec'].shape[0] == y.shape[0], "y size must match w_rec shape"
        assert params['w_rec'].shape[1] == y.shape[0], "y size must match w_rec shape"

    # using expected reward or just the reward:
    # 0 if expected_reward == reward
    reward_term = reward - expected_reward
    # reward_term = reward

    w_ff = update_layer_params(
        x, y, params['w_ff'], reward_term,  # theta['ff'] TODO
        theta, cfg.synapse_learning_rate['ff']
    )
    if "recurrent" in cfg.plasticity_layers:
        w_rec = update_layer_params(
            y, y, params['w_rec'], reward_term,  # theta['rec'] TODO
            theta, cfg.synapse_learning_rate['rec']
        )
    elif cfg.recurrent:
        w_rec = params['w_rec']
    else:
        w_rec = None

    params = {  # Always reassemble dict in jitted function to avoid in-place updates
        'w_ff': w_ff,
        'w_rec': w_rec,
        'w_out': params['w_out']  # No plasticity in output layer
    }

    return params

@partial(jax.jit, static_argnames=("plasticity_func", "cfg", "mode"))
def simulate_trajectory(
    key,
    input_params,
    init_params,
    ff_mask,
    rec_mask,
    theta,
    plasticity_func,
    exp_data,  # Data of one whole experiment, {(N_sessions, N_steps_per_session_max)}
    step_mask,
    cfg,
    mode
    ):
    """ Simulate trajectory of activations (and parameters) of one experiment (animal).

    Args:
        key: Random key for PRNG.
        input_params (N_inputs, N_hidden_pre): inputs --> presynaptic activations
        init_params (dict): w_ff: (num_hidden_pre, num_hidden_post),
                w_rec: (num_hidden_post, num_hidden_post) if recurrent,
                w_out: (num_hidden_post, 1)
                (b_ff, b_rec, b_out are not used for now)
        theta: Plasticity coefficients for the model.
        plasticity_func: Plasticity function to use.
        exp_data (dict): Data in {(N_sessions, N_steps_per_session_max, dim_element)}
        step_mask (N_sessions, N_steps_per_session_max): Valid (not padding) steps
        cfg: Configuration dictionary.
        mode:
            - 'simulation': return trajectories of x, y, output
            - 'generation_train': return x, y, output, decision, rewards
            - 'generation_test': return x, y, output, decision, rewards, params

    Returns:
        activity_trajec_exp {
            x_trajec_exp (N_sessions, N_steps_per_session_max, N_hidden_pre),
            y_trajec_exp (N_sessions, N_steps_per_session_max, N_hidden_post),
            output_trajec_exp (N_sessions, N_steps_per_session_max),
                # If 'generation' in mode:
            decision_trajec_exp (N_sessions, N_steps_per_session_max),
            reward_trajec_exp (N_sessions, N_steps_per_session_max),
            expected_reward_trajec_exp (N_sessions, N_steps_per_session_max),
                # If 'test' in mode:
            [params_trajec_exp {  # Only the plastic layers
                w_ff: (N_sessions, N_steps_per_session_max,
                       N_hidden_pre, N_hidden_post),
                w_rec: (N_sessions, N_steps_per_session_max,
                        N_hidden_post, N_hidden_post)
                }
            ]
        }: Trajectories of activations over the course of the experiment.
    """

    def simulate_session(params, session_variables):
        """ Simulate trajectory of parameters and activations within one session.

        Args:
            params: Initial parameters at the start of the session.
            session_variables: Tuple of (
                session_inputs, (N_steps_per_session_max,)
                session_rewards,
                session_expected_rewards,
                session_mask) for the session.

        Returns:
            params_session: Parameters at the end of the session.
            activity_trajec_session: {
                x_trajec_session (N_steps_per_session_max, N_hidden_pre),
                y_trajec_session (N_steps_per_session_max, N_hidden_post),
                output_trajec_session (N_steps_per_session_max),
                    # If 'generation' in mode:
                decision_trajec_session (N_steps_per_session_max),
                reward_trajec_session (N_steps_per_session_max),
                expected_reward_trajec_session (N_steps_per_session_max),
                    # If 'test' in mode:
                params_trajec_session: {  # Only the plastic layers
                    w_ff: (N_steps_per_session_max, N_hidden_pre, N_hidden_post),
                    w_rec: (N_steps_per_session_max, N_hidden_post, N_hidden_post),
                }
            }"""

        def simulate_step(params, step_variables):
            """ Simulate forward pass within one time step.

            Args:
                params (N_hidden_pre, N_hidden_post): carry for jax.lax.scan.
                step_variables: (
                    data: {inputs[, rewards, expected_rewards]}
                    valid: mask for this step (0 or 1)
                    step_key: Random key for PRNG.

            Returns:
                params: Updated parameters after the step.
                output_data: {x, y, output[,
                              decision, reward, expected_reward[, params]]}
            """
            input_data, valid, step_key = step_variables
            output_data = {}
            x, y, output = network_forward(step_key,
                                           input_params, params,
                                           ff_mask, rec_mask,
                                           input_data['inputs'], cfg)
            output_data['xs'] = x
            output_data['ys'] = y
            output_data['outputs'] = output

            # Allow python 'if' in jitted function because mode is static
            if 'generation' in mode:
                decision = compute_decision(step_key, output)
                # TODO reward is only temporarily probabilistic
                reward_key, _ = jax.random.split(step_key)
                reward = compute_reward(reward_key, decision)
                expected_reward = compute_expected_reward(reward, None)

                output_data['decisions'] = decision
                output_data['rewards'] = reward * cfg.reward_scale
                output_data['expected_rewards'] = expected_reward * cfg.reward_scale
            else:
                reward = input_data['rewards'] * cfg.reward_scale
                expected_reward = input_data['expected_rewards'] * cfg.reward_scale

            params = jax.lax.cond(valid,
                                  lambda p: update_params(
                                      x, y, params, reward, expected_reward,
                                      theta, plasticity_func, cfg),
                                  lambda p: p,
                                  params)

            if 'test' in mode:
                output_data['params'] = {}
                # Only return trajectories of plastic params
                if "feedforward" in cfg.plasticity_layers:
                    output_data['params']['w_ff'] = params['w_ff']
                if "recurrent" in cfg.plasticity_layers:
                    output_data['params']['w_rec'] = params['w_rec']
            return params, output_data

        # Run inner scan over steps within one session
        params_session, activity_trajec_session = jax.lax.scan(
            simulate_step, params, session_variables)

        return params_session, activity_trajec_session

    # Pre-split keys for each session and step
    n_sessions = step_mask.shape[0]
    n_steps = step_mask.shape[1]
    total_keys = int(n_sessions * n_steps)
    flat_keys = jax.random.split(key, total_keys + 1)[1:]
    session_step_keys = flat_keys.reshape((n_sessions, n_steps, flat_keys.shape[-1]))

    # Run outer scan over sessions
    _params_exp, activity_trajec_exp = jax.lax.scan(
        simulate_session,
        init_params,
        (exp_data,
         step_mask,
         session_step_keys)
    )

    return activity_trajec_exp
