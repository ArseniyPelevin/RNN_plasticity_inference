
import jax
import jax.numpy as jnp
import model
from scipy import stats
from utils import sample_truncated_normal


class Experiment:
    """Class to run a single experiment/animal/trajectory and handle generated data"""

    def __init__(self, key, exp_i, cfg, generation_theta, generation_func, mode):
        """Initialize experiment with given configuration and plasticity model.

        Args:
            key: JAX random key.
            exp_i: Experiment index.
            cfg: Configuration dictionary.
            generation_theta: 4D tensor of plasticity coefficients.
            generation_func: Function to compute plasticity.
            mode: "train" or "test".
        """

        self.exp_i = exp_i
        self.cfg = cfg
        self.generation_theta = generation_theta
        self.generation_func = generation_func
        self.data = {}

        # Generate random keys for different parts of the model
        (key,
         sessions_key,
         inputs_key,
         x_gen_key,
         x_train_key,
         ff_mask_key,
         rec_mask_key,
         weights_key,
         func_sparse_key,
         simulation_key) = jax.random.split(key, 10)

        # Pick random number of sessions in this experiment given mean and std
        num_sessions = sample_truncated_normal(
            sessions_key, cfg["mean_num_sessions"], cfg["sd_num_sessions"]
        )

        # Generate inputs and step mask for this experiment
        inputs, self.step_mask = self.generate_inputs(inputs_key, num_sessions)
        self.rewarded_pos = inputs['rewarded_pos']

        # Generate real presynaptic activity and don't save it - it is latent variable
        x_gen = self.generate_x(x_gen_key, inputs, mode='generation')
        # Generate assumed presynaptic activity and save for training
        x_train = self.generate_x(x_train_key, inputs, mode='training')
        self.data['x_train'] = x_train

        self.feedforward_mask_generation = self.generate_feedforward_mask(
            ff_mask_key, cfg["num_hidden_pre"], cfg["num_hidden_post"],
            cfg["feedforward_sparsity_generation"],
            cfg["postsynaptic_input_sparsity_generation"]
        )
        self.feedforward_mask_training = self.generate_feedforward_mask(
            ff_mask_key, cfg["num_hidden_pre"], cfg["num_hidden_post"],
            cfg["feedforward_sparsity_training"],
            cfg["postsynaptic_input_sparsity_training"]
        )

        self.recurrent_mask_generation = self.generate_recurrent_mask(
            rec_mask_key, cfg["num_hidden_post"], cfg["recurrent_sparsity_generation"],
            self.feedforward_mask_generation
        )
        self.recurrent_mask_training = self.generate_recurrent_mask(
            rec_mask_key, cfg["num_hidden_post"], cfg["recurrent_sparsity_training"],
            self.feedforward_mask_training
        )

        # num_hidden_pre -> num_hidden_post plasticity layer
        self.init_weights = model.initialize_weights(
            weights_key,
            cfg,
            cfg.init_weights_std_generation,
            cfg.init_weights_mean_generation
            )

        # Apply functional sparsity to plastic weights initialization during generation
        for layer in cfg.plasticity_layers:
            func_sparse_key, _ = jax.random.split(func_sparse_key)
            self.init_weights[f'w_{layer}'] *= jax.random.bernoulli(
                func_sparse_key,
                cfg.init_weights_sparsity_generation[layer],
                shape=self.init_weights[f'w_{layer}'].shape)

        trajectories = model.simulate_trajectory(simulation_key,
            self.init_weights,
            self.feedforward_mask_generation,
            self.recurrent_mask_generation,
            self.generation_theta,
            self.generation_func,
            x_gen,
            self.rewarded_pos,
            self.step_mask,
            cfg,
            mode=f'generation_{mode}'
        )
        self.data.update(trajectories)

        if mode == 'test':
            self.weights_trajec = self.data.pop('weights')

    def generate_inputs(self, key, num_sessions):

        # TODO? Is it a bad idea to use dict here?
        inputs = {}

        for session in range(num_sessions):
            key, n_trials_key, acdc_key = jax.random.split(key, 3)
            num_trials = sample_truncated_normal(
                n_trials_key,
                self.cfg["mean_trials_per_session"],
                self.cfg["sd_trials_per_session"])
            task_types = self.gen_2acdc(acdc_key, num_trials)
            for task_type in task_types:
                key, subkey = jax.random.split(key)

                if self.cfg["input_type"] == 'random':
                    trial_inputs = self.generate_random_trial_input(subkey)
                elif self.cfg["input_type"] == 'task':
                    trial_inputs = self.generate_task_trial_input(subkey, task_type)

                for var in trial_inputs:
                    (inputs.setdefault(var, [[] for _ in range(num_sessions)])[session]
                     .extend(trial_inputs[var]))

        return self.nested_inputs_lists_to_tensors(inputs)

    def nested_inputs_lists_to_tensors(self, inputs):
        """ Convert nested list of inputs per session to padded tensor and step mask.
        Args:
            inputs: per-input-variable dict of nested lists,
                outer list is over sessions, inner list is over time steps in session

        Returns:
            inputs_tensors: dict of arrays,
                shape (num_sessions, max_steps_per_session, var_dim)
            step_mask: array of shape (num_sessions, max_steps_per_session),
                1 for valid steps, 0 for padding
        """
        # Create step mask
        sample_input = inputs[list(inputs.keys())[0]]
        session_lengths = jnp.array([len(session) for session in sample_input])
        max_steps_per_session = jnp.max(session_lengths)
        step_mask = (jnp.arange(max_steps_per_session)[None, :]
                     < session_lengths[:, None]
                     ).astype(jnp.int32)

        # For each variable, convert nested list to padded tensor
        inputs_tensors = {}
        for var, var_input in inputs.items():
            # Create tensor and pad: (num_sessions, max_steps_per_session, var_dim)
            inputs_tensor = jnp.zeros((step_mask.shape[0], step_mask.shape[1],
                                       *var_input[0][0].shape))
            for s, session in enumerate(var_input):
                inputs_tensor = (inputs_tensor.at[s, :len(session)]
                                 .set(jnp.array(session)))
            inputs_tensors[var] = inputs_tensor

        return inputs_tensors, step_mask

    def gen_2acdc(self, key, n, lambd=0.7, max_rep=3):
        """ Generate a 2AFC sequence with Poisson-distributed repeats. """

        rep_key, first_key = jax.random.split(key)

        # Sample repeats (Poisson + 1, clipped to max_rep)
        reps = jax.random.poisson(rep_key, lambd, shape=(n,)).astype(jnp.int32) + 1
        reps = jnp.clip(reps, 1, max_rep)

        # Randomly choose first trial type and start alternating sequence
        t0 = jax.random.randint(first_key, (), 0, 2)  # First trial
        types = (t0 + jnp.arange(reps.shape[0])) % 2

        # Repeat trial types according to sampled repeats
        return jnp.repeat(types, reps)[:n]

    def generate_random_trial_input(self, key):
        """ Generate random input for one trial (Mehta et al., 2023).

        Returns:
            inputs: {'x' (num_steps, num_hidden_pre): array of presynaptic activity,
                     'rewarded_pos': (num_steps,) dummy to fit task input format}
        """
        key, n_steps_key = jax.random.split(key)
        # Configuration set specifically for random input regardless of time
        num_steps = sample_truncated_normal(n_steps_key,
                                            self.cfg["mean_steps_per_trial"],
                                            self.cfg["sd_steps_per_trial"])

        inputs = jnp.zeros((num_steps, self.cfg.num_hidden_pre))
        for step in range(num_steps):
            key, subkey = jax.random.split(key)
            step_input = jax.random.normal(subkey, shape=(self.cfg.num_hidden_pre,))
            step_input = (step_input * self.cfg.presynaptic_firing_std
                          + self.cfg.presynaptic_firing_mean)
            inputs = inputs.at[step].set(step_input)

        return {'x': inputs,
                'rewarded_pos': jnp.zeros((num_steps,))  # Dummy, not used
                }

    def generate_task_trial_input(self, key, trial_type):
        """ Generate structured task-based input for one trial (Sun et al., 2025).

        Args:
            key: JAX random key.
            trial_type: Integer indicating the type of trial (0 - near, 1 - far).

        Returns:
            inputs: {'t' (num_steps,): trial time in seconds,
                     'v' (num_steps,): velocity in cm/step,
                     'pos' (num_steps,): position in cm at each time step,
                     'cue' (num_steps,): visual cue type at each time step,
                     'rewarded_pos': (num_steps,) binary array of rewarded positions}
        """
        trial_time_key, v_pos_key = jax.random.split(key)
        trial_time = sample_truncated_normal(
            trial_time_key,
            mean=self.cfg.mean_trial_time,
            std=self.cfg.std_trial_time)

        # Generate velocity and position inputs
        t, v, pos = self.generate_velocity_and_position(v_pos_key, trial_time)

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

    def generate_velocity_and_position(self, key, trial_time):
        """ Generate velocity and position time series for one trial. """

        # Derived parameters
        num_steps = (trial_time - 2) / self.cfg.dt  # steps, minus 2s for teleportation
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

        # Integrate to get position, rescale to desired distance
        positions = jnp.cumsum(v_smooth)  # cm
        scale = self.cfg.trial_distance / positions[-1]
        v_smooth = v_smooth * scale
        positions = jnp.cumsum(v_smooth)  # cm

        # Add 2s of zero velocity and teleport to start (position is circular)
        position_at_teleport = jnp.ones(int(2/self.cfg.dt)) * self.cfg.trial_distance
        v_smooth = jnp.concatenate([v_smooth, jnp.zeros(int(2/self.cfg.dt))])
        positions = jnp.concatenate([positions, position_at_teleport])

        t = jnp.arange(0, trial_time, self.cfg.dt)

        return t, v_smooth, positions

    def generate_x(self, key, inputs, mode):
        """ Generate presynaptic activity based on input.

        Args:
            key: JAX random key.
            inputs: dict of input arrays.
            mode: 'generation' or 'training', adds variability in generation mode

        Returns:
            x: (n_sessions, n_steps, num_hidden_pre) presynaptic activity
        """
        if self.cfg["input_type"] == 'random':
            return inputs['x']  # Random input is already presynaptic activity
        elif self.cfg["input_type"] == 'task':
            # Positional presynaptic activity (n_sessions, n_steps, num_place_neurons)
            x_pos, _place_field_centers = self.generate_x_pos(key, inputs['pos'], mode)
            # Visual presynaptic activity (n_sessions, n_steps, num_visual_neurons)
            x_visual = jax.nn.one_hot(inputs['cue'],
                                      jnp.unique(inputs['cue']).shape[0])
            x_visual = x_visual.at[:,:,1:].get()  # No visual input at teleportation
            x_visual = x_visual.repeat(self.cfg.num_visual_neurons_per_type, axis=-1)
            # x_velocity = None  # TODO implement velocity input
            return jnp.concatenate([x_pos, x_visual], axis=-1)

    def generate_x_pos(self, key, positions, mode):
        """
        Generate presynaptic firing rates based on position using place fields.

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
        amplitudes = jnp.ones((self.cfg.num_place_neurons,)) \
            * self.cfg.place_field_amplitude_mean
        # Array of place field widths for each neuron
        place_field_widths = jnp.ones((self.cfg.num_place_neurons,)) \
            * self.cfg.place_field_width_mean

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

    def generate_feedforward_mask(self, key, n_pre, n_post,
                                  ff_sparsity, input_sparsity):
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

        if not self.cfg.recurrent:
            input_sparsity = 1.0

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

    def generate_recurrent_mask(self, key, n_post, rec_sparsity, ff_mask):
        """Generate a binary mask for the recurrent weights to enforce sparsity.

        Args:
            key: JAX random key.
            n_post: Number of postsynaptic neurons.
            rec_sparsity [0, 1]: Fraction of nonzero weights in the recurrent layer,
                all neurons are guaranteed to receive some input and some output:
                0 - at least one input per neuron,
                    not counting (allowed) autapses, but counting feedforward input,
                1 - all-to-all connectivity.

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
