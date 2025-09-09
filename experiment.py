
import jax
import jax.numpy as jnp
import model
from utils import sample_truncated_normal


class Experiment:
    """Class to run a single experiment/animal/trajectory and handle generated data"""

    def __init__(self, key, exp_i, cfg, plasticity_coeffs, plasticity_func, mode):
        """Initialize experiment with given configuration and plasticity model.

        Args:
            key: JAX random key.
            exp_i: Experiment index.
            cfg: Configuration dictionary.
            plasticity_coeffs: 4D tensor of plasticity coefficients.
            plasticity_func: Function to compute plasticity.
            mode: "train" or "test".
        """

        self.exp_i = exp_i
        self.cfg = cfg
        self.plasticity_coeffs = plasticity_coeffs
        self.plasticity_func = plasticity_func
        self.data = {}

        # Generate random keys for different parts of the model
        (key,
         sessions_key,
         inputs_key,
         input_params_key,
         ff_mask_key,
         rec_mask_key,
         params_key,
         simulation_key) = jax.random.split(key, 8)

        # Pick random number of sessions in this experiment given mean and std
        num_sessions = sample_truncated_normal(
            sessions_key, cfg["mean_num_sessions"], cfg["sd_num_sessions"]
        )

        (self.data['inputs'],
         self.step_mask  # (N_sessions, N_steps_per_session_max)
         ) = self.generate_inputs(inputs_key, num_sessions)

        # num_inputs -> num_hidden_pre embedding, fixed for one exp/animal
        self.input_params = model.initialize_input_parameters(  #TODO redefine?
            input_params_key,
            cfg["num_inputs"], cfg["num_hidden_pre"],
            input_params_scale=cfg["input_params_scale"]
        )

        self.feedforward_mask = self.generate_feedforward_mask(
            ff_mask_key, cfg["num_hidden_pre"], cfg["num_hidden_post"],
            cfg["feedforward_sparsity"], cfg["postsynaptic_input_sparsity"]
        )

        self.recurrent_mask = self.generate_recurrent_mask(
            rec_mask_key, cfg["num_hidden_post"], cfg["recurrent_sparsity"],
            self.feedforward_mask
        )

        # num_hidden_pre -> num_hidden_post plasticity layer
        self.init_params = model.initialize_parameters(params_key, cfg)

        trajectories = model.simulate_trajectory(simulation_key,
            self.input_params,
            self.init_params,
            self.feedforward_mask,  # if cfg.feedforward_sparsity < 1.0 else None,
            self.recurrent_mask,  # if cfg.recurrent_sparsity < 1.0 else None,
            plasticity_coeffs,
            plasticity_func,
            self.data,
            self.step_mask,
            cfg,
            mode=f'generation_{mode}'
        )
        self.data.update(trajectories)

        if mode == 'test':
            self.params_trajec = self.data.pop('params')

    def generate_inputs(self, key, num_sessions):
        def generate_input(key):
            # # Generate input - one integer out of number of classes num_inputs
            # step_input = jax.random.randint(key, shape=(1),
            #                                 minval=0, maxval=self.cfg.num_inputs)

            # Generate makeshift presynaptic input (TODO)
            step_input = jax.random.normal(key, shape=(self.cfg.num_hidden_pre,))
            step_input = (step_input * self.cfg.presynaptic_firing_std
                          + self.cfg.presynaptic_firing_mean)
            return step_input

        inputs = [[] for _ in range(num_sessions)]

        max_steps_per_session = 0
        for session in range(num_sessions):
            key, subkey = jax.random.split(key)
            num_trials = sample_truncated_normal(
                subkey,
                self.cfg["mean_trials_per_session"],
                self.cfg["sd_trials_per_session"])
            for _trial in range(num_trials):
                key, subkey = jax.random.split(subkey)
                num_steps = sample_truncated_normal(
                    subkey,
                    self.cfg["mean_steps_per_trial"],
                    self.cfg["sd_steps_per_trial"])
                for _step in range(num_steps):
                    key, subkey = jax.random.split(subkey)
                    step_input = generate_input(subkey)
                    inputs[session].append(step_input)
            max_steps_per_session = max(max_steps_per_session, len(inputs[session]))

        # Pad and convert to tensor
        inputs_tensor = jnp.zeros((num_sessions, max_steps_per_session,
                                   *inputs[0][0].shape))
        # Create mask of valid steps
        step_mask = jnp.zeros_like(inputs_tensor[..., 0])
        for s, session in enumerate(inputs):
            inputs_tensor = inputs_tensor.at[s, :len(session)].set(jnp.array(session))
            step_mask = step_mask.at[s, :len(session)].set(1)

        return inputs_tensor, step_mask

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
