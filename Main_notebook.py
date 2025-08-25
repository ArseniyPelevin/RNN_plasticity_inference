# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: plasticity_inference
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import importlib

import synapse


# +
config = {
    "num_inputs": 6,  # Number of input classes
    "num_hidden_pre": 100,  # x, presynaptic neurons for plasticity layer
    "num_hidden_post": 1000,  # y, postsynaptic neurons for plasticity layer
    "num_outputs": 1,  # m, binary decision (licking/not licking at this time step)
    "num_exp": 10,  # Number of experiments/trajectories
    # TODO account for different number of blocks within one experiment?
    "num_blocks": 15,  # Number of blocks/sessions/days per experiment/trajectory/animal
    # TODO account for different number of trials within one block?
    "num_trials_per_block": 100,  # Number of trials/runs in each block/session/day
    # TODO account for different number of steps within one trial?
    "num_steps_per_trial": 50,  # Number of sequential time steps in one trial/run
    "num_epochs": 250,
    "expid": 1,  # For saving results and seeding random
    "generation_plasticity": "1X1Y1W0R0-1X0Y2W1R0",  # Oja's rule
    "generation_model": "volterra",
    "plasticity_coeff_init": "random",
    "plasticity_model": "volterra",
}

cfg = OmegaConf.create(config)
# TODO cfg = validate_config(cfg)
key = jax.random.PRNGKey(cfg["expid"])

# +


def generate_experiments(key, cfg, plasticity_coeff, plasticity_func, mode):
    # Generate all experiments/trajectories
    # TODO differentiate num_train and num_eval
    for exp_i in range(cfg["num_exp"]):
        seed = (cfg.expid + 1) * (exp_i + 1)
        exp_key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(exp_key)  # TODO
        # num_inputs -> num_hidden_pre (6 -> 100) embedding, fixed for one exp/animal
        input_params = jax.random.normal(
            key, shape=(cfg["num_hidden_pre"], cfg["num_inputs"])
        )
        initial_params_scale = 0.01
        # num_hidden_pre -> num_hidden_post (100 -> 1000) plasticity layer
        params = (
            jax.random.normal(
                subkey, shape=(cfg["num_hidden_pre"], cfg["num_hidden_post"])
            )
            * initial_params_scale
        )
        generate_experiment(
            key, cfg, input_params, params, plasticity_coeff, plasticity_func, mode
        )


def generate_experiment(
    key, cfg, input_params, params, plasticity_coeff, plasticity_func, mode
):
    inputs, xs, ys, decisions, rewards, expected_rewards = (
        [[[] for _ in range(cfg.trials_per_block)] for _ in range(cfg.num_blocks)]
        for _ in range(6)  # Nested lists for each of the 6 variables
    )

    for block in range(cfg.num_blocks):
        for trial in range(cfg.trials_per_block):
            key, _ = jax.random.split(key)

            trial_data, params = generate_trial(
                key, cfg, input_params, params, plasticity_coeff, plasticity_func
            )
            (
                inputs[block][trial],
                xs[block][trial],
                ys[block][trial],
                decisions[block][trial],
                rewards[block][trial],
                expected_rewards[block][trial],
            ) = trial_data

    return inputs, xs, ys, decisions, rewards, expected_rewards


def generate_trial(key, cfg, input_params, params, plasticity_coeffs, plasticity_func):
    inputs, xs, ys, decisions, rewards, expected_rewards = [[] for _ in range(6)]
    # TODO manage all keys
    for step in range(cfg["num_steps_per_trial"]):
        input = jax.random.randint(key, (1), 0, cfg["num_inputs"])
        inputs.append(input)
        input_onehot = jax.nn.one_hot(input, cfg["num_inputs"])
        input_noise = jax.random.normal(key, (cfg["num_hidden_pre"],)) * 0.1
        x = jnp.dot(input_onehot, input_params) + input_noise
        xs.append(x)
        # Compute plasticity layer activity

    return (inputs, xs, ys, decisions, rewards, expected_rewards), params


# +
importlib.reload(synapse)

# Generate model activity
# TODO add branching for experimental data
generation_coeff, generation_func = synapse.init_plasticity(
    key, cfg, mode="generation_model"
)
data = generate_experiments(
    key, cfg, generation_coeff, generation_func, mode="generation"
)


# +


# Initialize parameters for training
# params, plasticity_coeff, plasticity_func, key = initialize_parameters(cfg, key)


# +


# Training

for epoch in range(config["num_epochs"]):
    # Simulate

    # Compute loss

    # Compute gradients

    # Update parameters

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        loss = []
        print(f"Epoch {epoch}, Loss: {loss}")
