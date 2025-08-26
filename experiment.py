from functools import partial

import jax
import jax.numpy as jnp
import mymodel
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
            input_key, shape=(cfg["num_inputs"], cfg["num_hidden_pre"])
        )
        # Standardize each input across classes
        self.input_params -= jnp.mean(self.input_params, axis=0, keepdims=True)
        self.input_params /= jnp.std(self.input_params, axis=0, keepdims=True) + 1e-8

        initial_params_scale = 0.01
        # num_hidden_pre -> num_hidden_post (100 -> 1000) plasticity layer
        self.params = (
            jax.random.normal(
                params_key, shape=(cfg["num_hidden_pre"], cfg["num_hidden_post"])
            )
            * initial_params_scale,  # w
            jnp.zeros((cfg["num_hidden_post"],)),  # b
        )
        # TODO loop inside a list for multilayer network

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
            [len(inputs[j][i]) for i in range(self.cfg.trials_per_session)]
            for j in range(self.cfg.num_sessions)
        ]
        max_trial_length = int(jnp.max(jnp.array(trial_lengths)))
        build_tensor = partial(experiment_list_to_tensor, max_trial_length)
        data = [
            build_tensor(var) for var in
            [inputs, xs, ys, decisions, rewards, expected_rewards]
        ]

        return data

    def generate_experiment(self, key):
        """Generate a synthetic experiment for a single animal/trajectory.
        Returns lists of trials for each variable.

        Args:
            key: JAX random key.

        Returns:
            (inputs,
            xs,
            ys,
            decisions,
            rewards,
            expected_rewards): Nested lists
                for each session
                for each trial within session
                of timeseries
        """
        inputs, xs, ys, decisions, rewards, expected_rewards = (
            [
                [[] for _ in range(self.cfg.trials_per_session)]
                for _ in range(self.cfg.num_sessions)
            ]
            for _ in range(6)  # Nested lists for each of the 6 variables
        )

        for session in range(self.cfg.num_sessions):
            for trial in range(self.cfg.trials_per_session):
                key, _ = jax.random.split(key)

                trial_data = self.generate_trial(
                    key, trial_length=self.cfg["num_steps_per_trial"])
                # In generation mode all trials have the same length(?)

                (
                    inputs[session][trial],
                    xs[session][trial],
                    ys[session][trial],
                    decisions[session][trial],
                    rewards[session][trial],
                    expected_rewards[session][trial],
                ) = trial_data

        return inputs, xs, ys, decisions, rewards, expected_rewards

    def generate_trial(self, key, trial_length):
        """Generate a timecourse of all variables for a single trial.

        Args:
            key: JAX random key.
            trial_length: Length of the trial.

        Returns:
            (inputs,
            xs,
            ys,
            decisions,
            rewards,
            expected_rewards): Arrays of the same length containing the timecourses
                for each variable. If the trial is shorter than the maximum length,
                it will be padded with zeros.

        """

        inputs, xs, ys, decisions, rewards, expected_rewards = ([] for _ in range(6))

        # TODO manage all keys
        for _step in range(trial_length):
            key, _ = jax.random.split(key)

            # Generate input
            new_input = jax.random.randint(key, (1), 0, self.cfg["num_inputs"])
            inputs.append(new_input)

            # Embed input into (hidden) presynaptic layer
            input_onehot = jax.nn.one_hot(new_input, self.cfg["num_inputs"]).squeeze()
            input_noise = jax.random.normal(key, (self.cfg["num_hidden_pre"],)) * 0.1
            x = jnp.dot(input_onehot, self.input_params) + input_noise

            xs.append(x)

            # Compute postsynaptic layer activity
            w, b = self.params
            y = jnp.dot(x, w) + b
            ys.append(y)

            # Compute decision
            output = jnp.mean(y)
            p_decision = jax.nn.sigmoid(output)
            decision = jax.random.bernoulli(key, p_decision).astype(float)
            decisions.append(decision)

            # TODO Compute reward
            reward = 0
            rewards.append(reward)
            expected_reward = reward
            expected_rewards.append(expected_reward)

            # Update parameters
            jit_update_params = partial(jax.jit, static_argnames=("plasticity_func",))(
                mymodel.update_params
            )
            self.params = jit_update_params(
                x,
                y,
                self.params,
                reward,
                expected_reward,
                self.plasticity_coeff,
                self.plasticity_func,
            )

        data = [jnp.array(var) for var in
                [inputs, xs, ys, decisions, rewards, expected_rewards]]

        return data
