import jax
import jax.numpy as jnp
import numpy as np
from utils import experiment_list_to_tensor


class Experiment:
    """Class to run a single experiment/animal/trajectory and handle generated data"""

    def __init__(self, exp_i, cfg, plasticity_coeff, plasticity_func):
        """Initialize experiment with given configuration and plasticity model.

        Args:
            exp_i: Experiment index.
            cfg: Configuration dictionary.
            plasticity_coeff: 4D tensor of plasticity coefficients.
            plasticity_func: Function to compute plasticity.
        """

        self.exp_i = exp_i
        self.cfg = cfg
        self.plasticity_coeff = plasticity_coeff
        self.plasticity_func = plasticity_func

        # Generate random keys for different parts of the model
        seed = (cfg["expid"] + 1) * (exp_i + 1)
        key = jax.random.PRNGKey(seed)
        input_key, params_key, exp_key = jax.random.split(key, 3)

        # num_inputs -> num_hidden_pre (6 -> 100) embedding, fixed for one exp/animal
        self.input_params = jax.random.normal(
            input_key, shape=(cfg["num_hidden_pre"], cfg["num_inputs"])
        )
        initial_params_scale = 0.01
        # num_hidden_pre -> num_hidden_post (100 -> 1000) plasticity layer
        self.params = (
            jax.random.normal(
                params_key, shape=(cfg["num_hidden_pre"], cfg["num_hidden_post"])
            )
            * initial_params_scale
        )

        data = self.generate_experiment(exp_key)
        data = self.experiment_lists_to_tensors(data)
        self.data = {
            "inputs": data[0],
            "xs": data[1],
            "ys": data[2],
            "decisions": data[3],
            "rewards": data[4],
            "expected_rewards": data[5],
        }

    def experiment_lists_to_tensors(self, data):
        """Convert lists of trials of different lengths to one trajectory tensor.

        Args:
            data: Dictionary of nested lists of trials for each variable.

        Returns:
            Tensor of the whole trajectory for each variable.
        """

        # TODO This function has a heavy dimensionality problem and does not work yet
        inputs, xs, ys, decisions, rewards, expected_rewards = data

        trial_lengths = [
            [len(inputs[j][i]) for i in range(self.cfg.trials_per_block)]
            for j in range(self.cfg.num_blocks)
        ]
        max_trial_length = np.max(np.array(trial_lengths))
        print(f"Exp {self.exp_i}, longest trial length: {max_trial_length}")

        print(f"Inputs shape 1: {len(inputs)}, {len(inputs[0])}, {len(inputs[0][0])}")
        inputs = experiment_list_to_tensor(max_trial_length, inputs, list_type="inputs")
        print(f"Inputs shape 2: {inputs.shape}")

        xs = experiment_list_to_tensor(max_trial_length, xs, list_type="xs")
        ys = experiment_list_to_tensor(max_trial_length, ys, list_type="ys")
        decisions = experiment_list_to_tensor(
            max_trial_length, decisions, list_type="decisions"
        )

        rewards = np.array(rewards, dtype=float).flatten()
        expected_rewards = np.array(expected_rewards, dtype=float).flatten()

        # print("odors: ", odors[exp_i])
        # print("rewards: ", rewards[exp_i])

        return inputs, xs, ys, decisions, rewards, expected_rewards

    def generate_experiment(self, key):
        """Generate a synthetic experiment for a single animal/trajectory.
        Returns lists of trials for each variable.

        Args:
            key: JAX random key.

        Returns:
            Nested lists of different length timeseries
                of each trials within each blocks
                for each variable.
        """
        inputs, xs, ys, decisions, rewards, expected_rewards = (
            [
                [[] for _ in range(self.cfg.trials_per_block)]
                for _ in range(self.cfg.num_blocks)
            ]
            for _ in range(6)  # Nested lists for each of the 6 variables
        )

        for block in range(self.cfg.num_blocks):
            for trial in range(self.cfg.trials_per_block):
                key, _ = jax.random.split(key)

                trial_data = self.generate_trial(key)

                (
                    inputs[block][trial],
                    xs[block][trial],
                    ys[block][trial],
                    decisions[block][trial],
                    rewards[block][trial],
                    expected_rewards[block][trial],
                ) = trial_data

        return inputs, xs, ys, decisions, rewards, expected_rewards

    def generate_trial(self, key):
        """Generate a timecourse of all variables for a single trial.

        Args:
            key: JAX random key.

        Returns:
            (inputs,
            xs,
            ys,
            decisions,
            rewards,
            expected_rewards): Lists of the same length containing the timecourses
                for each variable.

        """

        inputs, xs, ys, decisions, rewards, expected_rewards = ([] for _ in range(6))
        # TODO manage all keys
        for _step in range(self.cfg["num_steps_per_trial"]):
            key, _ = jax.random.split(key)

            # Generate input
            input_ = jax.random.randint(key, (1), 0, self.cfg["num_inputs"])
            inputs.append(input_)

            # Embed input into (hidden) presynaptic layer
            input_onehot = jax.nn.one_hot(input, self.cfg["num_inputs"]).squeeze()
            input_noise = jax.random.normal(key, (self.cfg["num_hidden_pre"],)) * 0.1
            x = jnp.dot(self.input_params, input_onehot) + input_noise
            xs.append(x)

            # TODO Compute postsynaptic layer activity

            # TODO Compute decision

            # TODO Compute reward

            # TODO Update parameters
            self.params = self.params

        return inputs, xs, ys, decisions, rewards, expected_rewards
