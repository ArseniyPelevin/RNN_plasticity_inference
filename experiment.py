
import jax
import jax.numpy as jnp
import mymodel
from utils import experiment_lists_to_tensors, sample_truncated_normal


class Experiment:
    """Class to run a single experiment/animal/trajectory and handle generated data"""

    def __init__(self, exp_i, cfg, plasticity_coeff, plasticity_func, num_sessions):
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
        key, input_params_key, params_key, exp_key = jax.random.split(key, 4)

        # num_inputs -> num_hidden_pre (6 -> 100) embedding, fixed for one exp/animal
        self.input_params = mymodel.initialize_input_parameters(
            input_params_key, cfg["num_inputs"], cfg["num_hidden_pre"]
        )

        # num_hidden_pre -> num_hidden_post (100 -> 1000) plasticity layer
        self.params = mymodel.initialize_parameters(
            params_key, cfg["num_hidden_pre"], cfg["num_hidden_post"]
        )

        data = self.generate_experiment(exp_key, num_sessions)
        data = experiment_lists_to_tensors(data)
        self.data = {
            "inputs": data[0],
            "xs": data[1],
            "ys": data[2],
            "decisions": data[3],
            "rewards": data[4],
            "expected_rewards": data[5],
        }

    def generate_experiment(self, key, num_sessions):
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
            [[] for _ in range(num_sessions)]
            for _ in range(6)
        )  # Nested lists for each of the 6 variables

        for session in range(num_sessions):
            key, subkey = jax.random.split(key)
            num_trials = sample_truncated_normal(
                subkey,
                self.cfg["mean_trials_per_session"],
                self.cfg["sd_trials_per_session"])
            for _trial in range(num_trials):
                key, sample_kay, trial_key = jax.random.split(key, 3)
                num_steps = sample_truncated_normal(
                    sample_kay,
                    self.cfg["mean_steps_per_trial"],
                    self.cfg["sd_steps_per_trial"])
                trial_data = self.generate_trial(
                    trial_key, num_steps)

                # trial_data should be a tuple/list of 6 items:
                (
                    trial_inputs,
                    trial_xs,
                    trial_ys,
                    trial_decisions,
                    trial_rewards,
                    trial_expected_rewards,
                ) = trial_data

                # append trial-level entries to this session's lists
                inputs[session].append(trial_inputs)
                xs[session].append(trial_xs)
                ys[session].append(trial_ys)
                decisions[session].append(trial_decisions)
                rewards[session].append(trial_rewards)
                expected_rewards[session].append(trial_expected_rewards)

        return inputs, xs, ys, decisions, rewards, expected_rewards

    def generate_trial(self, key, num_steps):
        """Generate a timecourse of all variables for a single trial.

        Args:
            key: JAX random key.
            num_steps: Length of the trial.

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
        for _step in range(num_steps):
            key, input_key, embed_key, decision_key = jax.random.split(key, 4)

            # Generate input
            new_input = jax.random.randint(input_key, (1), 0, self.cfg["num_inputs"])
            inputs.append(new_input)

            # Embed input into (hidden) presynaptic layer
            x = mymodel.embed_inputs_to_presynaptic(
                embed_key, new_input,
                self.cfg["num_inputs"], self.cfg["num_hidden_pre"],
                self.input_params)
            xs.append(x)

            # Compute postsynaptic layer activity
            y = mymodel.network_forward(x, self.params)
            ys.append(y)

            # Compute decision
            decision = mymodel.compute_decision(decision_key, y)
            decisions.append(decision)

            # TODO Compute reward
            reward = mymodel.compute_reward(decision)
            rewards.append(reward)
            expected_reward = reward
            expected_rewards.append(expected_reward)

            # Update parameters
            self.params = mymodel.update_params(
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
