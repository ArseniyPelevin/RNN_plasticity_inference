# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: plasticity_inference
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +

import importlib
from typing import Any  # TODO get rid of

import experiment
import jax
import losses
import model
import numpy as np
import optax
import synapse
from omegaconf import OmegaConf
from utils import sample_truncated_normal

# +
# coeff_mask = np.zeros((3, 3, 3, 3))
# coeff_mask[0:2, 0, 0, 0:2] = 1
coeff_mask = np.ones((3, 3, 3, 3))

config = {
    "num_inputs": 6,  # Number of input classes
    "num_hidden_pre": 100, # x, presynaptic neurons for plasticity layer
    "num_hidden_post": 1000,  # y, postsynaptic neurons for plasticity layer
    "num_outputs": 1,  # m, binary decision (licking/not licking at this time step)
    "num_exp": 1,  # Number of experiments/trajectories/animals

    "input_noise": 0.1,  # Standard deviation of noise added to presynaptic layer

    # Below commented are real values as per CA1 recording article. Be modest for now
    # "mean_num_sessions": 9,  # Number of sessions/days per experiment
    # "sd_num_sessions": 3,  # Standard deviation of sessions/days per experiment
    # "mean_trials_per_session": 124,  # Number of trials/runs in each session/day
    # "sd_trials_per_session": 43,  # Standard deviation of trials in each session/day
    # #TODO steps are seconds for now
    # "mean_steps_per_trial": 29,  # Number of sequential time steps in one trial/run
    # "sd_steps_per_trial": 10,  # Standard deviation of steps in each trial/run
    "mean_num_sessions": 5,  # Number of sessions/days per experiment/trajectory/animal
    "sd_num_sessions": 2,  # Standard deviation of sessions/days per animal
    "mean_trials_per_session": 7,  # Number of trials/runs in each session/day
    "sd_trials_per_session": 4,  # Standard deviation of trials in each session/day
    #TODO steps are seconds for now
    "mean_steps_per_trial": 50,  # Number of sequential time steps in one trial/run
    "sd_steps_per_trial": 10,  # Standard deviation of steps in each trial/run

    "num_epochs": 250,
    "expid": 1, # For saving results and seeding random
    "generation_plasticity": "1X1Y1W0R0-1X0Y2W1R0", # Oja's rule
    "generation_model": "volterra",
    "plasticity_coeff_init": "random",
    "plasticity_model": "volterra",

    # Restrictions on trainable plasticity parameters
    "trainable_coeffs": int(np.sum(coeff_mask)),
    "coeff_mask": coeff_mask.tolist(),
}

cfg = OmegaConf.create(config)
#TODO cfg = validate_config(cfg)
key = jax.random.PRNGKey(cfg["expid"])


# -

def generate_experiments(cfg, generation_coeff, generation_func, mode="generation"):
    # Generate all experiments/trajectories
    #TODO differentiate num_train and num_eval
    experiments = []
    # experiments_data = {}
    for exp_i in range(cfg['num_exp']):
        # Pick random number of sessions in this experiment given mean and std
        num_sessions = sample_truncated_normal(
            key, cfg["mean_num_sessions"], cfg["sd_num_sessions"]
        )
        exp = experiment.Experiment(exp_i, cfg,
                                    generation_coeff, generation_func,
                                    num_sessions)
        experiments.append(exp)
    return experiments


# +
importlib.reload(synapse)
importlib.reload(experiment)

# Generate model activity
#TODO add branching for experimental data
generation_coeff, generation_func = synapse.init_plasticity(
    key, cfg, mode="generation_model"
)
data = generate_experiments(
    cfg, generation_coeff, generation_func, mode="generation"
)
# -


for i in range(len(data)):
    print(data[i].data["inputs"].shape)

# +
importlib.reload(mymodel)

key, init_plasticity_key, init_params_key = jax.random.split(key, 3)
# Initialize parameters for training
generation_coeff, generation_func = synapse.init_plasticity(
    init_plasticity_key, cfg, mode="plasticity_model"
)
params = mymodel.initialize_parameters(
    init_params_key, cfg["num_hidden_pre"], cfg["num_hidden_post"]
)


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
