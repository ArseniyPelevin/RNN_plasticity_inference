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

import experiment
import jax
import synapse
from omegaconf import OmegaConf

# +
config = {
    "num_inputs": 6,  # Number of input classes
    "num_hidden_pre": 100, # x, presynaptic neurons for plasticity layer
    "num_hidden_post": 1000,  # y, postsynaptic neurons for plasticity layer
    "num_outputs": 1,  # m, binary decision (licking/not licking at this time step)
    "num_exp": 10,  # Number of experiments/trajectories
    #TODO account for different number of blocks within one experiment?
    "num_blocks": 5,  # Number of blocks/sessions/days per experiment/trajectory/animal
    #TODO account for different number of trials within one block?
    "trials_per_block": 12,  # Number of trials/runs in each block/session/day
    #TODO account for different number of steps within one trial?
    "num_steps_per_trial": 50,  # Number of sequential time steps in one trial/run
    "num_epochs": 250,
    "expid": 1, # For saving results and seeding random
    "generation_plasticity": "1X1Y1W0R0-1X0Y2W1R0", # Oja's rule
    "generation_model": "volterra",
    "plasticity_coeff_init": "random",
    "plasticity_model": "volterra"
}

cfg = OmegaConf.create(config)
#TODO cfg = validate_config(cfg)
key = jax.random.PRNGKey(cfg["expid"])


# -

def generate_experiments(cfg, generation_coeff, generation_func, mode="generation"):
    # Generate all experiments/trajectories
    #TODO differentiate num_train and num_eval
    experiments = []
    for exp_i in range(cfg['num_exp']):
        exp = experiment.Experiment(exp_i, cfg, generation_coeff, generation_func)
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
