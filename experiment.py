
from functools import partial

import jax
import jax.numpy as jnp
import model
from scipy import stats
from utils import sample_truncated_normal


def generate_experiments(key, cfg,
                         generation_theta, generation_func,
                         mode="train"):
    """ Generate all experiments/trajectories.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        cfg (dict): Configuration dictionary.
        generation_theta: 4D tensor of plasticity coefficients.
        generation_func: Plasticity function used for generation.
        mode: "train" or "test", decides if weight trajectories are returned.

    Returns:
        experiments (dict): Dictionary of arrays
            of shape (N_exp, ... ) for each variable of generated experiments.
    """

    if mode == "train":
        num_experiments = cfg.num_exp_train
    elif mode == "test":
        num_experiments = cfg.num_exp_test
    print(f"\nGenerating {num_experiments} {mode} trajectories")

    # Presplit keys for each experiment
    shapes_key, *experiment_keys = jax.random.split(key, num_experiments+1)

    # Define number of sessions, trials, steps for all experiments
    shapes, step_masks = define_experiments_shapes(shapes_key, num_experiments, cfg)

    # Build list of experiment dicts
    experiments_list = []
    for exp_i in range(num_experiments):
        exp = generate_experiment(
            experiment_keys[exp_i],
            exp_i, cfg,
            shapes, step_masks[exp_i],
            generation_theta, generation_func,
            mode
        )
        experiments_list.append(exp)
        print(f"Generated {mode} experiment {exp_i} with {shapes[0][exp_i]} sessions")

    # Aggregate into stacked arrays
    data_keys = list(experiments_list[0]['data'].keys())
    data = {k: jnp.array([e['data'][k] for e in experiments_list]) for k in data_keys}

    init_fixed_keys = list(experiments_list[0]['init_fixed_weights'].keys())
    init_fixed_weights = {
        k: jnp.array([e['init_fixed_weights'][k] for e in experiments_list])
        for k in init_fixed_keys
    }

    if 'weights_trajec' in experiments_list[0]:
        weights_trajec_keys = list(experiments_list[0]['weights_trajec'].keys())
        weights_trajec = {
            k: jnp.array([e['weights_trajec'][k] for e in experiments_list])
            for k in weights_trajec_keys}

    other_keys = ['step_mask', 'rewarded_pos',
                  'feedforward_mask_training', 'recurrent_mask_training',
                  'recording_mask', 'exp_i']
    agg_others = {k: jnp.array([e[k] for e in experiments_list]) for k in other_keys}

    # Final experiments dict with stacked arrays
    experiments = {
        'data': data,
        'init_fixed_weights': init_fixed_weights,
        **agg_others
    }
    if 'weights_trajec' in experiments_list[0]:
        experiments['weights_trajec'] = weights_trajec

    # # Debugging: print shapes of all arrays in experiments
    # for k, v in experiments['data'].items():
    #     print(f" data['{k}'] shape: {v.shape}")
    # for k, v in experiments['init_fixed_weights'].items():
    #     print(f" init_fixed_weights['{k}'] shape: {v.shape}")
    # if 'weights_trajec' in experiments:
    #     for k, v in experiments['weights_trajec'].items():
    #         print(f" weights_trajec['{k}'] shape: {v.shape}")
    # for k, v in experiments.items():
    #     if k != 'data' and k != 'init_fixed_weights' and k != 'weights_trajec':
    #         print(f"{k} shape: {v.shape}")

    return experiments

def define_experiments_shapes(key, num_exps, cfg):
    """ Define number of sessions, trials, and steps for all experiments.
    Also create step mask for all sessions in all experiments.

    Args:
        key: JAX random key.
        num_exps: Number of experiments.
        cfg: Configuration dictionary.

    Returns:
        shapes: Tuple of arrays:
            num_sessions: (num_experiments,),
            num_trials: (num_experiments, max_sessions),
            num_steps: (num_experiments, max_sessions, max_trials),
        step_mask: array (num_experiments, max_sessions, max_steps_per_session),
            1 for valid steps, 0 for padding.
    """
    sess_key, tr_key, st_key = jax.random.split(key, 3)

    # Define number of sessions in all experiments given mean and std
    num_sessions = sample_truncated_normal(
        sess_key,
        cfg["mean_num_sessions"], cfg["std_num_sessions"],
        num_exps
    )
    max_sessions = int(num_sessions.max())

    # Define number of trials in all sessions of all experiments
    num_trials = sample_truncated_normal(
        tr_key,
        cfg["mean_trials_per_session"], cfg["std_trials_per_session"],
        (num_exps, max_sessions)
    )
    sessions_idx = jnp.arange(max_sessions)[None, :]  # (1, max_sessions)
    sessions_mask = sessions_idx < num_sessions[:, None]  # (num_exps, max_sessions)
    num_trials = num_trials * sessions_mask  # Zero for nonexistent sessions
    max_trials = int(num_trials.max())

    # Define number of steps in all trials of all sessions of all experiments
    num_steps = sample_truncated_normal(
        st_key,
        cfg["mean_steps_per_trial"], cfg["std_steps_per_trial"],
        (num_exps, max_sessions, max_trials)
    )
    trials_idx = jnp.arange(max_trials)[None, None, :]  # (1,1,max_trials)
    trial_mask = trials_idx < num_trials[:, :, None]  # (num_exps, max_sess, max_trials)
    num_steps = num_steps * trial_mask  # Zero for nonexistent trials

    # Create step mask for all sessions in all experiments
    steps_per_session = jnp.sum(num_steps, axis=2)  # (num_experiments, max_sessions)
    max_steps_per_session = int(steps_per_session.max())  # scalar
    # (num_experiments, max_sessions, max_steps_per_session)
    step_mask = (jnp.arange(max_steps_per_session)[None, None, :]  # (1,1,max_steps)
                 < steps_per_session[:, :, None])

    return (num_sessions, num_trials, num_steps), step_mask.astype(jnp.int32)

def generate_experiment(key, exp_i, cfg, shapes, step_mask,
                generation_theta, generation_func, mode):
    """Initialize experiment with given configuration and plasticity model.

    Args:
        key: JAX random key.
        exp_i: Experiment index.
        cfg: Configuration dictionary.
        shapes: Tuple of (num_sessions, num_trials, num_steps) arrays,
        generation_theta: 4D tensor of plasticity coefficients.
        generation_func: Plasticity function used for generation.
        mode: "train" or "test", decides if weight trajectories are returned.
    """
    exp = {}
    exp['data'] = {}
    exp['exp_i'] = jnp.array(exp_i)
    exp['step_mask'] = step_mask

    # Generate random keys for different parts of the model
    (key,
     inputs_key,
     x_gen_key,
     x_train_key,
     ff_mask_key,
     rec_mask_key,
     weights_key,
     fixed_weights_key,
     func_sparse_key,
     simulation_key) = jax.random.split(key, 10)

    # Generate inputs and step mask for this experiment
    inputs = generate_inputs(inputs_key, shapes, step_mask, cfg, exp_i)
    exp['rewarded_pos'] = inputs['rewarded_pos']

    # Generate real presynaptic activity and don't save it - it is latent variable
    x_gen = generate_x(x_gen_key, inputs, cfg, mode='generation')
    # Generate assumed presynaptic activity and save for training
    x_train = generate_x(x_train_key, inputs, cfg, mode='training')
    exp['data']['x_train'] = x_train

    feedforward_mask_generation = generate_feedforward_mask(
        ff_mask_key, cfg["num_hidden_pre"], cfg["num_hidden_post"],
        cfg["feedforward_sparsity_generation"],
        cfg["postsynaptic_input_sparsity_generation"] if cfg.recurrent else 1.0
    )
    exp['feedforward_mask_training'] = generate_feedforward_mask(
        ff_mask_key, cfg["num_hidden_pre"], cfg["num_hidden_post"],
        cfg["feedforward_sparsity_training"],
        cfg["postsynaptic_input_sparsity_training"] if cfg.recurrent else 1.0
    )

    recurrent_mask_generation = generate_recurrent_mask(
        rec_mask_key, cfg["num_hidden_post"], cfg["recurrent_sparsity_generation"],
        feedforward_mask_generation
    )
    exp['recurrent_mask_training'] = generate_recurrent_mask(
        rec_mask_key, cfg["num_hidden_post"], cfg["recurrent_sparsity_training"],
        exp['feedforward_mask_training']
    )

    exp['recording_mask'] = jax.random.bernoulli(
        rec_mask_key, cfg["neural_recording_sparsity"], cfg["num_hidden_post"]
        ).astype(jnp.float32)

    # Initialize weights of all layers for generation of this experiment
    exp['init_weights'] = model.initialize_weights(
        weights_key,
        cfg,
        cfg.init_weights_std_generation,
        cfg.init_weights_mean_generation
        )

    # Initialize weights of fixed (non-trainable) layers for training of this experiment
    exp['init_fixed_weights'] = model.initialize_weights(
        fixed_weights_key,
        cfg,
        cfg.init_weights_std_training,
        layers = [layer for layer in ['w_ff', 'w_rec', 'w_out']
                  if layer not in cfg.trainable_init_weights]
    )

    # Apply functional sparsity to plastic weights initialization during generation
    func_sparse_keys = jax.random.split(func_sparse_key, len(cfg.plasticity_layers))
    for i, layer in enumerate(cfg.plasticity_layers):
        exp['init_weights'][f'w_{layer}'] *= jax.random.bernoulli(
            func_sparse_keys[i],
            cfg.init_weights_sparsity_generation[layer],
            shape=exp['init_weights'][f'w_{layer}'].shape)

    trajectories = model.simulate_trajectory(simulation_key,
        exp['init_weights'],
        feedforward_mask_generation,
        recurrent_mask_generation,
        generation_theta,
        generation_func,
        x_gen,
        exp['rewarded_pos'],
        exp['step_mask'],
        cfg,
        mode=f'generation_{mode}'
    )
    exp['data'].update(trajectories)

    if mode == 'test':
        exp['weights_trajec'] = exp['data'].pop('weights')

    return exp

def generate_inputs(key, shapes, step_mask, cfg, exp_i):
    """ Generate inputs for all sessions in one experiment.

    Args:
        key: JAX random key.
        shapes: Tuple of (num_sessions, num_trials, num_steps) arrays.
        step_mask: array of shape (num_sessions, max_steps_per_session),
        cfg: Configuration dictionary.
        exp_i: Experiment index.

    Returns:
        inputs_tensors: dict of arrays,
            shape (num_sessions, max_steps_per_session, var_dim)
        step_mask: array of shape (num_sessions, max_steps_per_session),
            1 for valid steps, 0 for padding
    """
    num_sessions, num_trials, num_steps = shapes

    num_sessions_ = num_sessions[exp_i]  # In this experiment
    max_trials_ = num_trials[exp_i].max()  # Across sessions in this experiment

    # Presplit keys for each session and trial
    acdc_key, trial_key = jax.random.split(key)
    acdc_rep_keys, acdc_first_keys = jax.random.split(acdc_key, 2 * num_sessions_
                                                      ).reshape(2, num_sessions_, 2)
    trial_keys = jax.random.split(trial_key, num_sessions_ * max_trials_
                                  ).reshape(num_sessions_, max_trials_, 2)

    inputs = {}
    for session_i in range(num_sessions_):
        num_trials_ = num_trials[exp_i, session_i]
        # Generate 2ACDC task sequence with Poisson-distributed repeats
        task_types = gen_2acdc((acdc_rep_keys[session_i],
                                acdc_first_keys[session_i]), num_trials_)
        for task_i, task_type in enumerate(task_types):
            num_steps_ = num_steps[exp_i, session_i, task_i]
            # Generate inputs for one trial
            if cfg["input_type"] == 'random':
                trial_inputs = generate_random_trial_input(
                    trial_keys[session_i, task_i], num_steps_, cfg)
            elif cfg["input_type"] == 'task':
                trial_inputs = generate_task_trial_input(
                    trial_keys[session_i, task_i], num_steps_, cfg, task_type)

            # Append trial inputs to session inputs
            for var in trial_inputs:
                (inputs.setdefault(var, [[] for _ in range(num_sessions_)]
                                   )[session_i]
                                   .extend(trial_inputs[var]))

    return nested_input_lists_to_tensors(inputs, step_mask.shape[0], step_mask.shape[1])

def nested_input_lists_to_tensors(inputs, max_sessions, max_steps):
    """ Convert nested list of inputs per session to padded tensor.
    Args:
        inputs: per-input-variable dict of nested lists,
            outer list is over sessions, inner list is over time steps in session
        max_sessions: Maximum number of sessions across all experiments.
        max_steps: Maximum number of steps per session across all experiments.

    Returns:
        inputs_tensors: dict of arrays,
            shape (max_sessions, max_steps_per_session_across_exps, *var_dim)
    """
    # For each variable, convert nested list to padded tensor
    inputs_tensors = {}
    for var, var_input in inputs.items():
        # Create tensor and pad: (num_sessions, max_steps_per_session, var_dim)
        inputs_tensor = jnp.zeros((max_sessions, max_steps,
                                   *var_input[0][0].shape))
        for s, session in enumerate(var_input):
            inputs_tensor = (inputs_tensor.at[s, :len(session)]
                             .set(jnp.array(session)))
        inputs_tensors[var] = inputs_tensor

    return inputs_tensors

def gen_2acdc(keys, n, lambd=0.7, max_rep=3):
    """ Generate a 2AFC sequence of length n with Poisson-distributed repeats. """

    rep_key, first_key = keys

    # Sample repeats (Poisson + 1, clipped to max_rep)
    reps = jax.random.poisson(rep_key, lambd, shape=(n,)).astype(jnp.int32) + 1
    reps = jnp.clip(reps, 1, max_rep)

    # Randomly choose first trial type and start alternating sequence
    t0 = jax.random.randint(first_key, (), 0, 2)  # First trial
    types = (t0 + jnp.arange(reps.shape[0])) % 2

    # Repeat trial types according to sampled repeats
    return jnp.repeat(types, reps)[:n]

def generate_random_trial_input(key, num_steps, cfg):
    """ Generate random input for one trial (Mehta et al., 2023).

    Returns:
        inputs: {'x' (num_steps, num_hidden_pre): array of presynaptic activity,
                 'rewarded_pos': (num_steps,) dummy to fit task input format}
    """
    x = jax.random.normal(key, shape=(num_steps, cfg.num_hidden_pre))
    x = x * cfg.presynaptic_firing_std + cfg.presynaptic_firing_mean

    return {'x': x,
            'rewarded_pos': jnp.zeros((num_steps,))  # Dummy, not used
            }

def generate_task_trial_input(key, num_steps, cfg, trial_type):
    """ Generate structured task-based input for one trial (Sun et al., 2025).

    Args:
        key: JAX random key.
        num_steps: Number of time steps in the trial.
        cfg: Configuration dictionary.
        trial_type: Integer indicating the type of trial (0 - near, 1 - far).

    Returns:
        inputs: {'t' (num_steps,): trial time in seconds,
                 'v' (num_steps,): velocity in cm/step,
                 'pos' (num_steps,): position in cm at each time step,
                 'cue' (num_steps,): visual cue type at each time step,
                 'rewarded_pos': (num_steps,) binary array of rewarded positions}
    """

    # Generate velocity and position inputs
    t, v, pos = generate_velocity_and_position(key, num_steps, cfg)

    # Generate visual cue sequence
    # [1,1,1,1,1,1,2,2,2,2,1,1,1,4,4,1,1,1,5,5,1,1,1,0,0,0]
    visual_cue_seq = [jnp.repeat(1, 6),
                      jnp.repeat(2, 4) + trial_type,  # Indicator
                      jnp.repeat(1, 3),
                      jnp.repeat(4, 2),  # Reward near
                      jnp.repeat(1, 3),
                      jnp.repeat(5, 2),  # Reward far
                      jnp.repeat(1, 3),
                      jnp.repeat(0, 3),  # Teleportation
                      ]
    visual_type_seq = jnp.concatenate(visual_cue_seq)

    # Compute segment index from continuous position (floor of x/10)
    segment_at_time = jnp.floor(pos / 10.0).astype(jnp.int32)
    # Choose visual cue in the current segment
    cue_at_time = visual_type_seq[segment_at_time]

    # Define rewarded positions along the trial length based on trial type
    rewarded_position = jnp.zeros_like(pos)
    if trial_type == 0:
        rewarded_position = jnp.where(cue_at_time == 4, 1.0, 0.0)
    elif trial_type == 1:
        rewarded_position = jnp.where(cue_at_time == 5, 1.0, 0.0)

    return {'t': t,  # Careful, after concatenating trials, time is not continuous
            'v': v,
            'pos': pos,
            'cue': cue_at_time,
            'rewarded_pos': rewarded_position}

def generate_velocity_and_position(key, num_steps, cfg):
    """ Generate velocity and position time series for one trial. """

    # Derived parameters
    num_steps = num_steps - 2 / cfg.dt  # steps, minus 2s for teleportation
    v_mean = cfg.trial_distance / num_steps  # cm/dt
    v_window = int(cfg.velocity_smoothing_window / cfg.dt)  # steps
    num_steps = int(num_steps)

    # Generate raw velocity signal and smooth it
    v = jax.random.normal(key, (num_steps,))
    gaussian_filter = stats.norm.pdf(jnp.linspace(-3, 3, v_window))
    gaussian_filter /= jnp.sum(gaussian_filter)
    v_smooth = jnp.convolve(v, gaussian_filter, mode='same')

    # Rescale to desired mean and std
    target_velocity_std = cfg.velocity_std * cfg.dt  # cm/s -> cm/dt
    observed_velocity_std = jnp.std(v_smooth)
    v_smooth = v_smooth * target_velocity_std / (observed_velocity_std + 1e-12)
    v_smooth = v_smooth + v_mean  # cm/dt

    # Integrate to get position, rescale to desired distance
    positions = jnp.cumsum(v_smooth)  # cm
    scale = cfg.trial_distance / positions[-1]
    v_smooth = v_smooth * scale
    positions = jnp.cumsum(v_smooth)  # cm

    # Add 2s of zero velocity and teleport to start (position is circular)
    position_at_teleport = jnp.ones(int(2/cfg.dt)) * cfg.trial_distance
    v_smooth = jnp.concatenate([v_smooth, jnp.zeros(int(2/cfg.dt))])
    positions = jnp.concatenate([positions, position_at_teleport])

    t = jnp.arange(0, num_steps * cfg.dt, cfg.dt)

    return t, v_smooth, positions

@partial(jax.jit, static_argnames=["cfg", "mode"])
def generate_x(key, inputs, cfg, mode):
    """ Generate presynaptic activity based on input.

    Args:
        key: JAX random key.
        inputs: dict of input arrays.
        cfg: Configuration dictionary.
        mode: 'generation' or 'training', adds variability in generation mode

    Returns:
        x: (n_sessions, n_steps, num_hidden_pre) presynaptic activity
    """
    if cfg["input_type"] == 'random':
        return inputs['x']  # Random input is already presynaptic activity

    elif cfg["input_type"] == 'task':
        # Positional presynaptic activity (n_sessions, n_steps, num_place_neurons)
        x_pos, _place_field_centers = generate_x_pos(key, inputs['pos'], cfg, mode)
        # Visual presynaptic activity (n_sessions, n_steps, num_visual_neurons)
        num_visual_types = 6  # Including teleportation
        x_visual = jax.nn.one_hot(inputs['cue'],
                                  num_visual_types)
        x_visual = x_visual.repeat(cfg.num_visual_neurons_per_type, axis=-1)
        x_velocity = inputs['v'][:, :, None].repeat(cfg.num_velocity_neurons, axis=-1)
        x_velocity /= jnp.mean(inputs['v'])  # Velocity firing represents relative speed
        return jnp.concatenate([x_pos, x_visual, x_velocity], axis=-1)

def generate_x_pos(key, positions, cfg, mode):
    """
    Generate presynaptic firing rates based on position using place fields.

    Args:
        key: JAX random key.
        positions: (n_sessions, n_steps) Array of positions at each time step in cm
        cfg: Configuration dictionary.
        mode: 'generation' or 'training', adds variability in generation mode

    Returns:
        rates: (n_sessions, n_steps, num_place_neurons)
        place_field_centers: (num_place_neurons,)
    """
    # Arrays of place field centers for each neuron
    place_field_centers = jnp.linspace(0, cfg.trial_distance, cfg.num_place_neurons)
    # Array of peak firing rates for each neuron
    amplitudes = jnp.ones((cfg.num_place_neurons,)) * cfg.place_field_amplitude_mean
    # Array of place field widths for each neuron
    place_field_widths = jnp.ones((cfg.num_place_neurons,)) * cfg.place_field_width_mean

    # Add latent variability to place field parameters for generation
    if mode == 'generation':
        centers_key, amp_key, width_key = jax.random.split(key, 3)
        # Add some jitter to place field centers for generation
        place_field_centers += jax.random.normal(centers_key, (cfg.num_place_neurons,)
                                                 ) * cfg.place_field_center_jitter
        # Add some jitter to amplitudes for generation
        amplitudes += jax.random.normal(amp_key, (cfg.num_place_neurons,)
                                        ) * cfg.place_field_amplitude_std
        amplitudes = jnp.clip(amplitudes,
                              a_min=0.0)  # avoid negative maxima
        # Add some jitter to widths for generation
        place_field_widths += jax.random.normal(width_key, (cfg.num_place_neurons,)
                                                ) * cfg.place_field_width_std
        place_field_widths = jnp.clip(place_field_widths,
                                      a_min=0.0)  # avoid negative widths

    # Convert linear variables to circular
    theta = 2 * jnp.pi * positions / cfg.trial_distance
    mu = 2 * jnp.pi * place_field_centers / cfg.trial_distance
    ang_sigma = 2 * jnp.pi * place_field_widths / cfg.trial_distance

    # Compute firing rates using von Mises function
    dtheta = theta[..., None] - mu[None, :]
    kappa = 1.0 / (ang_sigma**2 + 1e-12)
    vonMises = jnp.exp(kappa * (jnp.cos(dtheta) - 1.0))

    rates = vonMises * amplitudes[None, None, :]

    return rates, place_field_centers

def generate_feedforward_mask(key, n_pre, n_post, ff_sparsity, input_sparsity):
    """Generate a binary mask for the feedforward weights to enforce sparsity.

    Args:
        key: JAX random key.
        n_pre: Number of presynaptic neurons.
        n_post: Number of postsynaptic neurons.
        ff_sparsity [0, 1]: Fraction of nonzero weights in the feedforward layer,
            of all postsynaptic neurons receiving input (input_sparsity),
            all presynaptic neurons are guaranteed to have some output:
            0 - max(n_pre, n_post * input_sparsity) nonzero weights,
            1 - all presynaptic are connected to all input-receiving postsynaptic.
        input_sparsity [0, 1]: Fraction of postsynaptic neurons receiving input.

    Returns:
        A binary mask of shape (n_pre, n_post).
    """
    col_key, mask_key, fill_col_key, fill_row_key = jax.random.split(key, 4)

    # Choose input postsynaptic neurons (input columns)
    n_input_post = max(1, int(round(input_sparsity * n_post)))
    input_cols = jax.random.choice(
        col_key, n_post, shape=(n_input_post,), replace=False
    )

    # Generate random mask with given sparsity
    mask = jnp.zeros((n_pre, n_post))
    bern = jax.random.bernoulli(mask_key, p=float(ff_sparsity),
                                shape=(n_pre, n_input_post)
                                ).astype(jnp.float32)
    mask = mask.at[:, input_cols].set(bern)

    # Ensure no zero rows (presynaptic neurons without output)
    row_sums = mask[:, input_cols].sum(axis=1)
    zero_rows = jnp.where(row_sums == 0)[0]
    chosen_cols = jax.random.choice(
        fill_col_key, input_cols, shape=zero_rows.shape, replace=True
    )
    mask = mask.at[zero_rows, chosen_cols].set(1)

    # Ensure no zero selected columns (input postsynaptic neurons without input)
    col_sums = mask[:, input_cols].sum(axis=0)
    zero_cols_idx = jnp.where(col_sums == 0)[0]
    zero_cols = input_cols[zero_cols_idx]
    chosen_rows = jax.random.choice(
        fill_row_key, n_pre, shape=zero_cols.shape, replace=True
    )
    mask = mask.at[chosen_rows, zero_cols].set(1)

    return mask

def generate_recurrent_mask(key, n_post, rec_sparsity, ff_mask):
    """Generate a binary mask for the recurrent weights to enforce sparsity.

    Args:
        key: JAX random key.
        n_post: Number of postsynaptic neurons.
        rec_sparsity [0, 1]: Fraction of nonzero weights in the recurrent layer,
            all neurons are guaranteed to receive some input and some output:
            0 - at least one input per neuron,
                not counting (allowed) autapses, but counting feedforward inputs,
            1 - all-to-all connectivity.
        ff_mask: Feedforward mask (n_pre, n_post) to count feedforward inputs.

    Returns:
        A binary mask of shape (n_post, n_post).
    """
    mask_key, fill_key = jax.random.split(key)

    # Generate random mask with given sparsity
    mask = jax.random.bernoulli(mask_key, p=float(rec_sparsity),
                                shape=(n_post, n_post)
                                ).astype(jnp.float32)

    # Construct test mask to ensure at least one input per neuron
    # Autapses are not counted as input
    test_mask = mask.at[jnp.diag_indices(n_post)].set(0)
    # Feedforward input is counted as input
    test_mask = jnp.vstack([test_mask, ff_mask.sum(axis=0)])

    # Ensure no zero columns in test mask (postsynaptic neurons without any input)
    col_sums = test_mask.sum(axis=0)
    zero_cols = jnp.where(col_sums == 0)[0]
    chosen_rows = jax.random.choice(
        fill_key, n_post, shape=zero_cols.shape, replace=True
    )
    # If any diagonal elements were chosen, shift them up by one
    diagonal = jnp.where(zero_cols == chosen_rows)[0]
    chosen_rows = chosen_rows.at[diagonal].subtract(1)

    mask = mask.at[chosen_rows, zero_cols].set(1)

    return mask
