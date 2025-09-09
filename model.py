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

def initialize_parameters(key,
                          num_pre, num_post,
                          init_params_scale, plasticity_layers):
    """Initialize parameters for the network.

    num_hidden_pre -> num_hidden_post (100 -> 1000) plasticity layer

    Args:
        key: JAX random key.
        num_pre: Number of presynaptic neurons.
        num_post: Number of postsynaptic neurons.
        init_params_scale: Scale for initializing parameters.
        plasticity_layers: List of strings, "feedforward" and/or "recurrent".

    Returns:
        params: Tuple of (weights, biases) for the layer.
    """

    if "feedforward" in plasticity_layers and "recurrent" in plasticity_layers:
        num_pre = num_pre + num_post
    elif "feedforward" in plasticity_layers:
        num_pre = num_pre
    elif "recurrent" in plasticity_layers:
        num_pre = num_post
    else:
        raise ValueError("'feedforward' or 'recurrent' must be in plasticity_layers")

    # Use ""Xavier normal"" (paper's Kaiming)
    if init_params_scale == 'Xavier':
        init_params_scale = 1 / jnp.sqrt(num_pre + num_post)
    # TODO loop inside a list for multilayer network
    weights = (
        jax.random.normal(key, shape=(num_pre, num_post))
        * init_params_scale
    )
    biases = jnp.zeros((num_post,))
    return weights, biases

@partial(jax.jit, static_argnames=("cfg"))
def network_forward(key, input_params, params, ff_mask, rec_mask, step_input, cfg):
    """ Propagate through all layers from input to output. """

    # # Embed input integer into presynaptic layer activity
    # input_onehot = jax.nn.one_hot(step_input, cfg.num_inputs).squeeze()
    # x = jnp.dot(input_onehot, input_params)

    # Makeshift for input firing (TODO)
    x = step_input

    input_noise = jax.random.normal(key, (cfg.num_hidden_pre,))
    x += input_noise * cfg.presynaptic_noise_std

    # Forward pass through plastic layer. x -- params --> y
    w, b = params

    pre = None
    # Plastic feedforward layer
    if "feedforward" in cfg.plasticity_layers:
        pre = x
        ff_w = w[:cfg.num_hidden_pre]
    # Non-plastic feedforward layer
    else:
        ff_w = jnp.ones((cfg.num_hidden_pre, cfg.num_hidden_post))
        # Scale ff weights if no plasticity: by constant scale and by number of inputs
        ff_w = ff_w * cfg.feedforward_input_scale / ff_mask.sum(axis=0)
    ff_w *= ff_mask  # Apply feedforward sparsity mask
    y = x @ ff_w  # + b

    if cfg.recurrent:
        # Plastic recurrent layer
        if "recurrent" in cfg.plasticity_layers:
            pre = jnp.vstack([pre, y]) if pre else y
            rec_w = w[-cfg.num_hidden_post:]
        # Non-plastic recurrent layer
        else:
            rec_w = jnp.ones((cfg.num_hidden_post, cfg.num_hidden_post))
            # Scale rec weights if no plasticity: by const scale and by num of inputs
            rec_w = rec_w * cfg.recurrent_input_scale / rec_mask.sum(axis=0)
        rec_w *= rec_mask
        y += y @ rec_w  # + b

    # Add nonlinearity
    y = jax.nn.sigmoid(y)

    # Compute output probability ((1,) logit) based on postsynaptic layer activity
    output = jnp.mean(y)
    output = jax.nn.sigmoid(output)

    return x, y, output, pre

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
    pre, post, params,
    reward, expected_reward,
    plasticity_coeffs, plasticity_func,
    cfg
):
    """
    Update parameters in one layer.

    Calculates full weight matrix regardless of sparsity mask(s).
    Args:
        #!!! Function redefined from initial to update only one layer!
        pre (array): Either x for feedforward, or y for recurrent, or both stacked
        post (array): y (postsynaptic activations)
        params (list): Tuple (weights, biases) for one layer.
        reward (float): Reward at this timestep. TODO Not implemented
        expected_reward (float): Expected reward at this timestep.
        plasticity_coeffs (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        cfg (dict): Configuration dictionary.

    Returns: Updated parameters.
    """

    # Will only be checked at compile time
    assert pre.ndim == 1, "Input must be a vector"
    assert post.ndim == 1, "Output must be a vector"
    assert len(params) == 2, "Params must be a tuple (weights, biases)"
    assert params[0].shape[0] == pre.shape[0], "Input size must match weight shape"
    assert params[0].shape[1] == post.shape[0], "Output size must match weight shape"
    assert params[1].shape[0] == post.shape[0], "Output size must match bias shape"

    # using expected reward or just the reward:
    # 0 if expected_reward == reward
    reward_term = reward - expected_reward
    # reward_term = reward

    w, b = params

    # Allow python 'if' in jitted function because cfg is static
    # Use vectorized volterra_plasticity_function
    if cfg.plasticity_model == "volterra":
        dw = plasticity_func(pre, post, w, reward_term, plasticity_coeffs)
    # Use per-synapse mlp_plasticity_function
    elif cfg.plasticity_model == "mlp":
        # vmap over postsynaptic neurons
        vmap_post = jax.vmap(plasticity_func, in_axes=(None, 0, 0, None, None))
        # vmap over presynaptic neurons now, i.e. together vmap over synapses
        vmap_synapses = jax.vmap(vmap_post, in_axes=(0, None, 0, None, None))
        dw = vmap_synapses(pre, post, w, reward_term, plasticity_coeffs)

    # TODO decide whether to update bias or not
    db = jnp.zeros_like(b)
    # db = vmap_post(1.0, reward_term, b, plasticity_coeffs)

    assert (
        dw.shape == w.shape and db.shape == b.shape
    ), "dw and w should be of the same shape to prevent broadcasting \
        while adding"

    lr = cfg.synapse_learning_rate # / x.shape[0]
    params = (utils.softclip(w + lr * dw, cap=cfg.synaptic_weight_threshold),
              b + lr * db
              )  # TODO rewrite as list comprehension for multilayer

    return params

@partial(jax.jit, static_argnames=("plasticity_func", "cfg", "mode"))
def simulate_trajectory(
    key,
    input_params,
    init_params,
    ff_mask,
    rec_mask,
    plasticity_coeffs,
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
        init_params tuple((N_hidden_pre, N_hidden_post) - w
                          (N_hidden_post) - b
                         ): Initial synaptic weights and biases.
        plasticity_coeffs: Plasticity coefficients for the model.
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
            [params_trajec_exp (
                (N_sessions, N_steps_per_session_max, N_hidden_pre, N_hidden_post), # w
                (N_sessions, N_steps_per_session_max, N_hidden_post)  # b
                )]
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
                [params_trajec_session (
                    (N_steps_per_session_max, N_hidden_pre, N_hidden_post),  # w
                    (N_steps_per_session_max, N_hidden_post)  # b
                    )
                ]
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
            x, y, output, _pre = network_forward(step_key,
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
                                      _pre, y, params, reward, expected_reward,
                                      plasticity_coeffs, plasticity_func, cfg),
                                  lambda p: p,
                                  params)

            if 'test' in mode:
                output_data['params'] = params

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
