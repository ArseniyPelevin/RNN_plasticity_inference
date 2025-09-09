
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
         params_key,
         simulation_key) = jax.random.split(key, 6)

        # Pick random number of sessions in this experiment given mean and std
        num_sessions = sample_truncated_normal(
            sessions_key, cfg["mean_num_sessions"], cfg["sd_num_sessions"]
        )

        (self.data['inputs'],
         self.step_mask  # (N_sessions, N_steps_per_session_max)
         ) = self.generate_inputs(inputs_key, num_sessions)

        # num_inputs -> num_hidden_pre embedding, fixed for one exp/animal
        self.input_params = model.initialize_input_parameters(
            input_params_key,
            cfg["num_inputs"], cfg["num_hidden_pre"],
            input_params_scale=cfg["input_params_scale"]
        )

        # num_hidden_pre -> num_hidden_post plasticity layer
        init_params = model.initialize_parameters(
            params_key,
            cfg["num_hidden_pre"], cfg["num_hidden_post"],
            cfg["init_params_scale"]
        )

        trajectories = model.simulate_trajectory(simulation_key,
            self.input_params, init_params,
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
            step_input = (step_input * self.cfg.input_firing_std
                          + self.cfg.input_firing_mean)
            return step_input

        inputs = [[] for _ in range(num_sessions)]
        step_mask = [[] for _ in range(num_sessions)]

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
                step_mask[session] += [1] * num_steps
                for _step in range(num_steps):
                    key, subkey = jax.random.split(subkey)
                    step_input = generate_input(subkey)
                    inputs[session].append(step_input)
            max_steps_per_session = max(max_steps_per_session, len(inputs[session]))

        # Pad and convert to tensor
        inputs_tensor = jnp.zeros((num_sessions, max_steps_per_session,
                                   *inputs[0][0].shape))
        for s, session in enumerate(inputs):
            inputs_tensor = inputs_tensor.at[s, :len(session)].set(jnp.array(session))

        return inputs_tensor, jnp.array(step_mask)

