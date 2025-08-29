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
import utils
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
    "num_exp_train": 10,  # Number of experiments/trajectories/animals
    "num_exp_eval": 5,

    "input_firing_mean": 0.75,
    "input_firing_std": 0.01,  # Standard deviation of noise added to presynaptic layer
    "learning_rate": 1e-3,

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
    "mean_steps_per_trial": 20,  # Number of sequential time steps in one trial/run
    "sd_steps_per_trial": 5,  # Standard deviation of steps in each trial/run

    "num_epochs": 250,
    "expid": 1, # For saving results and seeding random
    "generation_plasticity": "1X1Y1W0R0-1X0Y2W1R0", # Oja's rule
    "generation_model": "volterra",
    "plasticity_coeffs_init": "random",
    "plasticity_model": "volterra",

    "regularization_type": "l1",
    "regularization_scale": 1e-2,

        # if "neural" in cfg.fit_data:
        # neural_loss = neural_mse_loss(  # TODO Look into and update
        #     key,
        #     mask,
        #     cfg.neural_recording_sparsity,
        #     cfg.measurement_noise_scale,

    "fit_data": "neural",
    "neural_recording_sparsity": 1,
    "measurement_noise_scale": 0,

    # Restrictions on trainable plasticity parameters
    "trainable_coeffs": int(np.sum(coeff_mask)),
    "coeff_mask": coeff_mask.tolist(),
}

cfg = OmegaConf.create(config)
#TODO cfg = validate_config(cfg)
key = jax.random.PRNGKey(cfg["expid"])


# -

def generate_experiments(key, cfg,
                         generation_coeff, generation_func,
                         mode="generation"):
    # Generate all experiments/trajectories
    if mode == "train":
        num_experiments = cfg.num_exp_train
        print(f"\nGenerating {num_experiments} trajectories")
    else:
        num_experiments = cfg.num_exp_eval
        print(f"\nGenerating {num_experiments} trajectories")

    experiments = []
    # experiments_data = {}
    for exp_i in range(num_experiments):
        # Pick random number of sessions in this experiment given mean and std
        key, num_sessions = sample_truncated_normal(
            key, cfg["mean_num_sessions"], cfg["sd_num_sessions"]
        )
        exp = experiment.Experiment(exp_i, cfg,
                                    generation_coeff, generation_func,
                                    num_sessions)
        experiments.append(exp)
        # experiments_data[exp_i] = exp.data
        print(f"Generated experiment {exp_i} with {num_sessions} sessions")

    return key, experiments


# +
importlib.reload(synapse)
importlib.reload(experiment)
importlib.reload(model)
importlib.reload(losses)
importlib.reload(utils)

# Generate model activity
key, subkey = jax.random.split(key)
#TODO add branching for experimental data
generation_coeff, generation_func = synapse.init_plasticity(
    subkey, cfg, mode="generation_model"
)
key, experiments = generate_experiments(
    key, cfg, generation_coeff, generation_func, mode="generation"
)
# -


print(len(experiments[0].data))
for exp in experiments:
    print(exp.data["inputs"].shape)
    print(exp.steps_per_session)
    print(type(exp.input_params))


# +
# importlib.reload(model)

key, init_plasticity_key, init_params_key = jax.random.split(key, 3)
# Initialize parameters for training
plasticity_coeffs, plasticity_func = synapse.init_plasticity(
    init_plasticity_key, cfg, mode="plasticity_model"
)

print(f'{type(plasticity_coeffs)=}, {plasticity_coeffs.shape=}')

#TODO
# !!! Parameters should probably NOT be initialized just once!
# They should be reassigned for each epoch and exp!
initial_params = model.initialize_parameters(
    init_params_key, cfg["num_hidden_pre"], cfg["num_hidden_post"]
)


# +
# importlib.reload(synapse)
# importlib.reload(experiment)
# importlib.reload(model)
# importlib.reload(losses)
# importlib.reload(utils)

# Return value (scalar) of the function (loss value)
# and gradient wrt its parameter at argnum (plasticity_coeffs)
loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=3) # !Check argnums!
optimizer = optax.adam(learning_rate=cfg["learning_rate"])
opt_state = optimizer.init(plasticity_coeffs)
expdata: dict[str, Any] = {}
# resampled_xs, neural_recordings, decisions, rewards, expected_rewards = data

losses_list = []

for epoch in range(cfg["num_epochs"] + 1):
    for exp in experiments:
        key, subkey = jax.random.split(key)
        loss, meta_grads = loss_value_and_grad(
            subkey,  # Pass subkey this time, because loss will not return key
            exp.input_params,
            initial_params,  # TODO update for each epoch/experiment
            plasticity_coeffs,  # Current plasticity coeffs, updated on each iteration
            plasticity_func,  # Static within losses
            exp.data['inputs'],
            # exp.data['xs'],  # Don't need it, will recompute based on input_parameters
            exp.data['ys'],
            exp.data['decisions'],
            exp.data['rewards'],
            exp.data['expected_rewards'],
            exp.mask,
            cfg,  # Static within losses
        )
        updates, opt_state = optimizer.update(
            meta_grads, opt_state, plasticity_coeffs
        )
        plasticity_coeffs = optax.apply_updates(plasticity_coeffs, updates)

    if epoch % cfg["logging_interval"] == 0:
        print(f'{epoch=}, {loss=}')
        losses_list.append(loss)
print(plasticity_coeffs)


# -



