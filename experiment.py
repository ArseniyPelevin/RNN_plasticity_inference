
import equinox as eqx
import jax
import jax.numpy as jnp
import network
import plasticity
import simulation
from scipy import stats
from utils import sample_truncated_normal


def generate_experiments(key, cfg, mode, num_exps=None):
    """ Generate all experiments/trajectories.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        cfg (dict): Configuration dictionary of experiment parameters.
        mode: "train" or "test", decides if weight trajectories are returned.

    Returns:
        experiments (list): List of Experiment objects.
    """
    if num_exps is None:
        if mode == "train":
            num_exps = cfg.experiment.num_exp_train
        elif mode == "test":
            num_exps = cfg.experiment.num_exp_test
    print(f"\nGenerating {num_exps} {mode} trajectories")

    # Presplit keys for each experiment
    (plasticity_gen_key,
     w_init_key,
     shapes_key,
     *experiment_keys) = jax.random.split(key, num_exps + 3)

    # Define number of sessions, trials, steps for all experiments
    shapes, step_masks = define_experiments_shapes(shapes_key, num_exps, cfg.experiment)

    # Initialize plasticity for generation
    plasticity_gen = plasticity.initialize_plasticity(
        plasticity_gen_key, cfg.plasticity, mode='generation', init_scale=0.0)

    # Build list of experiment dicts
    experiments_list = []
    for exp_i in range(num_exps):
        exp = Experiment(
            experiment_keys[exp_i],
            exp_i,
            shapes, step_masks[exp_i],
            plasticity_gen,
            cfg,
            mode
        )
        experiments_list.append(exp)
        print(f"Generated {mode} experiment {exp_i} with {shapes[0][exp_i]} sessions")

    return experiments_list

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
        step_mask (bool): array (num_experiments, max_sessions, max_steps_per_session)
    """
    sess_key, tr_key, st_key = jax.random.split(key, 3)

    # Define number of sessions in all experiments given mean and std
    num_sessions = sample_truncated_normal(
        sess_key,
        cfg.mean_num_sessions, cfg.std_num_sessions,
        num_exps
    )
    max_sessions = int(num_sessions.max())

    # Define number of trials in all sessions of all experiments
    num_trials = sample_truncated_normal(
        tr_key,
        cfg.mean_trials_per_session, cfg.std_trials_per_session,
        (num_exps, max_sessions)
    )
    sessions_idx = jnp.arange(max_sessions)[None, :]  # (1, max_sessions)
    sessions_mask = sessions_idx < num_sessions[:, None]  # (num_exps, max_sessions)
    num_trials = num_trials * sessions_mask  # Zero for nonexistent sessions
    max_trials = int(num_trials.max())

    # Define number of steps in all trials of all sessions of all experiments
    num_steps = sample_truncated_normal(
        st_key,
        cfg.mean_steps_per_trial, cfg.std_steps_per_trial,
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

    return (num_sessions, num_trials, num_steps), step_mask

class Experiment(eqx.Module):
    exp_i: int
    cfg: object = eqx.field(static=True)
    network: eqx.Module
    w_init_gen: jnp.array
    w_init_train: jnp.array
    step_mask: jnp.array
    x_input: jnp.array
    rewarded_pos: jnp.array
    data: dict
    weights_trajec: dict = None  # Only for test experiments

    def __init__(self, key, exp_i, shapes, step_mask,
                 plasticity_gen, cfg, mode):
        """Initialize experiment with given configuration and plasticity model.

        Args:
            key: JAX random key.
            exp_i: Experiment index.
            shapes (tuple): (num_sessions, num_trials, num_steps) arrays,
            step_mask (bool): array of shape (num_sessions, max_steps_per_session),
            generation_plasticity: per-plastic-layer dict of eqx Plasticity instances.
            training_plasticity: per-plastic-layer dict of eqx Plasticity instances.
            cfg: Configuration dictionary of experiment parameters.
            mode: "train" or "test", decides if weight trajectories are returned.
        """
        self.exp_i = exp_i
        self.step_mask = step_mask
        self.cfg = cfg.experiment
        self.data = {}

        # Generate random keys for different parts of the model
        (net_gen_key,
         net_train_key,
         w_gen_key,
         w_train_key,
         inputs_key,
         x_gen_key,
         x_train_key,
         simulation_key) = jax.random.split(key, 8)

        # Initialize two different networks with different sparsity masks.
        # Generation network is not saved, only its activity is used.
        # Training network is saved and used for training.
        network_gen = network.Network(net_gen_key, cfg.network,
                                      self.cfg.input_type, mode='generation')
        if not cfg.training.same_init_connectivity:
            self.network = network.Network(net_train_key, cfg.network,
                                        self.cfg.input_type, mode='training')
        else:
            self.network = network_gen

        # Initialize weights for generation and training. Store them to experiment
        self.w_init_gen = network.initialize_weights(w_gen_key, cfg.network)
        if not cfg.training.same_init_weights:
            self.w_init_train = network.initialize_weights(w_train_key, cfg.network)
        else:
            self.w_init_train = self.w_init_gen  # Use same initial weights for training

        # Apply initial weights to corresponding networks
        network_gen = network_gen.apply_weights(self.w_init_gen)
        self.network = self.network.apply_weights(self.w_init_train)

        # Generate inputs and step mask for this experiment
        inputs = self.generate_inputs(inputs_key, shapes)
        self.rewarded_pos = inputs['rewarded_pos']

        # Generate real input activity and don't save it - it is latent variable
        x_gen = self.generate_x(x_gen_key, inputs, mode='generation')
        # Generate assumed input activity and save for training or testing
        if not cfg.training.same_input:
            x_train = self.generate_x(x_train_key, inputs, mode='training')
        else:
            x_train = x_gen
        self.x_input = x_train

        trajectories = simulation.simulate_trajectory(
            simulation_key,
            self,
            x_gen,
            network_gen,
            plasticity_gen,
            returns=('xs', 'ys', 'outputs', 'decisions', 'rewards',
                     'weights' if mode == 'test' else '')
        )
        self.data.update(trajectories)

        if mode == 'test':
            self.weights_trajec = self.data.pop('weights')

    def generate_inputs(self, key, shapes):
        """ Generate inputs for all sessions in one experiment.

        Args:
            key: JAX random key,
            shapes (tuple): (num_sessions, num_trials, num_steps) arrays.

        Returns:
            inputs_tensors: dict of arrays,
                shape (num_sessions, max_steps_per_session, var_dim)
            step_mask: array of shape (num_sessions, max_steps_per_session),
                1 for valid steps, 0 for padding
        """
        num_sessions, num_trials, num_steps = shapes

        num_sessions_ = int(num_sessions[self.exp_i])  # In this experiment
        max_trials_ = int(num_trials[self.exp_i].max())  # Across sessions in this exp

        # Presplit keys for each session and trial
        acdc_key, trial_key = jax.random.split(key)

        acdc_rep_keys, acdc_first_keys = jax.random.split(
            acdc_key, 2 * num_sessions_
            ).reshape((2, num_sessions_) + acdc_key.shape)
        trial_keys = jax.random.split(
            trial_key, num_sessions_ * max_trials_
            ).reshape((num_sessions_, max_trials_) + trial_key.shape)

        inputs = {}
        for session_i in range(num_sessions_):
            num_trials_ = num_trials[self.exp_i, session_i]
            # Generate 2ACDC task sequence with Poisson-distributed repeats
            task_types = self.gen_2acdc((acdc_rep_keys[session_i],
                                         acdc_first_keys[session_i]), num_trials_)
            for task_i, task_type in enumerate(task_types):
                num_steps_ = num_steps[self.exp_i, session_i, task_i]
                # Generate inputs for one trial
                if self.cfg.input_type == 'random':
                    trial_inputs = self.generate_random_trial_input(
                        trial_keys[session_i, task_i], num_steps_)
                elif self.cfg.input_type == 'task':
                    trial_inputs = self.generate_task_trial_input(
                        trial_keys[session_i, task_i], num_steps_, task_type)

                # Append trial inputs to session inputs
                for var in trial_inputs:
                    (inputs.setdefault(var, [[] for _ in range(num_sessions_)]
                                    )[session_i]
                                    .extend(trial_inputs[var]))

        max_sessions, max_steps = self.step_mask.shape[0], self.step_mask.shape[1]
        return self.nested_input_lists_to_tensors(inputs, max_sessions, max_steps)

    def nested_input_lists_to_tensors(self, inputs, max_sessions, max_steps):
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

    def gen_2acdc(self, keys, n, lambd=0.7, max_rep=3):
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

    def generate_random_trial_input(self, key, num_steps):
        """ Generate random input for one trial (Mehta et al., 2023).

        Returns:
            inputs: {'x' (num_steps, num_x_neurons): array of input activity,
                     'rewarded_pos': (num_steps,) dummy to fit task input format}
        """
        x = jax.random.normal(key, shape=(num_steps, self.network.cfg.num_x_neurons))
        x = x * self.cfg.input_firing_std + self.cfg.input_firing_mean

        return {'x': x,
                'rewarded_pos': jnp.zeros((num_steps,))  # Dummy, not used
                }

    def generate_task_trial_input(self, key, num_steps, trial_type):
        """ Generate structured task-based input for one trial (Sun et al., 2025).

        Args:
            key: JAX random key.
            num_steps: Number of time steps in the trial.
            trial_type: Integer indicating the type of trial (0 - near, 1 - far).

        Returns:
            inputs: {'t' (num_steps,): trial time in seconds,
                     'v' (num_steps,): velocity in cm/step,
                     'pos' (num_steps,): position in cm at each time step,
                     'cue' (num_steps,): visual cue type at each time step,
                     'rewarded_pos': (num_steps,) binary array of rewarded positions}
        """

        # Generate velocity and position inputs
        t, v, pos = self.generate_velocity_and_position(key, num_steps)

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
            rewarded_position = jnp.where(cue_at_time == 4, True, False)
        elif trial_type == 1:
            rewarded_position = jnp.where(cue_at_time == 5, True, False)

        return {'t': t,  # Careful, after concatenating trials, time is not continuous
                'v': v,
                'pos': pos,
                'cue': cue_at_time,
                'rewarded_pos': rewarded_position}

    def generate_velocity_and_position(self, key, num_steps):
        """ Generate velocity and position time series for one trial. """

        # Derived parameters
        num_steps = num_steps - 2 / self.cfg.dt  # steps, minus 2s for teleportation
        v_mean = self.cfg.trial_distance / num_steps  # cm/dt
        v_window = int(self.cfg.velocity_smoothing_window / self.cfg.dt)  # steps
        num_steps = int(num_steps)

        # Generate raw velocity signal and smooth it
        v = jax.random.normal(key, (num_steps,))
        gaussian_filter = stats.norm.pdf(jnp.linspace(-3, 3, v_window))
        gaussian_filter /= jnp.sum(gaussian_filter)
        v_smooth = jnp.convolve(v, gaussian_filter, mode='same')

        # Rescale to desired mean and std
        target_velocity_std = self.cfg.velocity_std * self.cfg.dt  # cm/s -> cm/dt
        observed_velocity_std = jnp.std(v_smooth)
        v_smooth = v_smooth * target_velocity_std / (observed_velocity_std + 1e-12)
        v_smooth = v_smooth + v_mean  # cm/dt
        v_smooth = jnp.clip(v_smooth, a_min=0.0)  # No negative velocities

        # Integrate to get position, rescale to desired distance
        positions = jnp.cumsum(v_smooth)  # cm
        scale = self.cfg.trial_distance / positions[-1]
        v_smooth = v_smooth * scale
        positions = jnp.cumsum(v_smooth)  # cm

        # Add 2s of zero velocity and teleport to start (position is circular)
        position_at_teleport = jnp.ones(int(2/self.cfg.dt)) * self.cfg.trial_distance
        v_smooth = jnp.concatenate([v_smooth, jnp.zeros(int(2/self.cfg.dt))])
        positions = jnp.concatenate([positions, position_at_teleport])

        t = jnp.arange(0, num_steps * self.cfg.dt, self.cfg.dt)

        return t, v_smooth, positions

    @eqx.filter_jit
    def generate_x(self, key, inputs, mode):
        """ Generate X layer activity based on input.

        Args:
            key: JAX random key.
            inputs: dict of input arrays.
            mode: 'generation' or 'training', adds variability in generation mode

        Returns:
            x: (n_sessions, n_steps, num_x_neurons) X layer activity
        """
        if self.cfg.input_type == 'random':
            return inputs['x']  # Random input is already X layer activity

        elif self.cfg.input_type == 'task':
            # Positional input activity (n_sessions, n_steps, num_place_neurons)
            x_pos, _place_field_centers = self.generate_x_pos(key, inputs['pos'], mode)
            # Visual input activity (n_sessions, n_steps, num_visual_neurons)
            num_visual_types = 6  # Including teleportation
            x_visual = jax.nn.one_hot(inputs['cue'],
                                      num_visual_types)
            x_visual = x_visual.repeat(self.cfg.num_visual_neurons_per_type, axis=-1)
            x_velocity = inputs['v'][:, :, None].repeat(
                self.cfg.num_velocity_neurons, axis=-1)
            # Velocity firing represents relative speed
            x_velocity /= jnp.mean(inputs['v'])
            return jnp.concatenate([x_pos, x_visual, x_velocity], axis=-1)

    def generate_x_pos(self, key, positions, mode):
        """
        Generate X layer activity based on position using place fields.

        Args:
            key: JAX random key.
            positions: (n_sessions, n_steps) Array of positions at each time step in cm
            mode: 'generation' or 'training', adds variability in generation mode

        Returns:
            rates: (n_sessions, n_steps, num_place_neurons)
            place_field_centers: (num_place_neurons,)
        """
        # Arrays of place field centers for each neuron
        place_field_centers = jnp.linspace(0, self.cfg.trial_distance,
                                           self.cfg.num_place_neurons)
        # Array of peak firing rates for each neuron
        amplitudes = jnp.ones((self.cfg.num_place_neurons,)
                              ) * self.cfg.place_field_amplitude_mean
        # Array of place field widths for each neuron
        place_field_widths = jnp.ones((self.cfg.num_place_neurons,)
                                      ) * self.cfg.place_field_width_mean

        # Add latent variability to place field parameters for generation
        if mode == 'generation':
            centers_key, amp_key, width_key = jax.random.split(key, 3)
            # Add some jitter to place field centers for generation
            place_field_centers += jax.random.normal(
                centers_key, (self.cfg.num_place_neurons,)
                ) * self.cfg.place_field_center_jitter
            # Add some jitter to amplitudes for generation
            amplitudes += jax.random.normal(
                amp_key, (self.cfg.num_place_neurons,)
                ) * self.cfg.place_field_amplitude_std
            amplitudes = jnp.clip(amplitudes,
                                  a_min=0.0)  # avoid negative maxima
            # Add some jitter to widths for generation
            place_field_widths += jax.random.normal(
                width_key, (self.cfg.num_place_neurons,)
                ) * self.cfg.place_field_width_std
            place_field_widths = jnp.clip(place_field_widths,
                                          a_min=0.0)  # avoid negative widths

        # Convert linear variables to circular
        theta = 2 * jnp.pi * positions / self.cfg.trial_distance
        mu = 2 * jnp.pi * place_field_centers / self.cfg.trial_distance
        ang_sigma = 2 * jnp.pi * place_field_widths / self.cfg.trial_distance

        # Compute firing rates using von Mises function
        dtheta = theta[..., None] - mu[None, :]
        kappa = 1.0 / (ang_sigma**2 + 1e-12)
        vonMises = jnp.exp(kappa * (jnp.cos(dtheta) - 1.0))

        rates = vonMises * amplitudes[None, None, :]

        return rates, place_field_centers
