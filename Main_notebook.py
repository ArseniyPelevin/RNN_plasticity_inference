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

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import training
from omegaconf import OmegaConf

# +
# coeff_mask = np.zeros((3, 3, 3, 3))
# coeff_mask[0:2, 0, 0, 0:2] = 1
coeff_mask = np.ones((3, 3, 3, 3))
coeff_mask[:, :, :, 1:] = 0  # Zero out reward coefficients

config = {
    "use_experimental_data": False,

    "num_inputs": 1000,  # Number of input classes (num_epochs * 4 for random normal)
    "num_hidden_pre": 100, # x, presynaptic neurons for plasticity layer
    "num_hidden_post": 1000,  # y, postsynaptic neurons for plasticity layer
    "num_outputs": 1,  # m, binary decision (licking/not licking at this time step)
    "num_exp_train": 50,  # Number of experiments/trajectories/animals
    "num_exp_eval": 0,

    "input_firing_mean": 0,
    "input_firing_std": 1,  # Standard deviation of input firing rates
    "input_noise_std": 0,  # Standard deviation of noise added to presynaptic layer
    "synapse_learning_rate": 0.1,
    "learning_rate": 3e-3,

    "input_params_scale": 1,
    "initial_params_scale": 0.1,  # float or 'Xavier'

    # Below commented are real values as per CA1 recording article. Be modest for now
    # "mean_num_sessions": 9,  # Number of sessions/days per experiment
    # "sd_num_sessions": 3,  # Standard deviation of sessions/days per experiment
    # "mean_trials_per_session": 124,  # Number of trials/runs in each session/day
    # "sd_trials_per_session": 43,  # Standard deviation of trials in each session/day
    # #TODO steps are seconds for now
    # "mean_steps_per_trial": 29,  # Number of sequential time steps in one trial/run
    # "sd_steps_per_trial": 10,  # Standard deviation of steps in each trial/run
    "mean_num_sessions": 1,  # Number of sessions/days per experiment/trajectory/animal
    "sd_num_sessions": 0,  # Standard deviation of sessions/days per animal
    "mean_trials_per_session": 1,  # Number of trials/runs in each session/day
    "sd_trials_per_session": 0,  # Standard deviation of trials in each session/day
    #TODO steps are seconds for now
    "mean_steps_per_trial": 50,  # Number of sequential time steps in one trial/run
    "sd_steps_per_trial": 0,  # Standard deviation of steps in each trial/run

    "num_epochs": 250,
    "expid": 10, # For saving results and seeding random

    "generation_plasticity": "1X1Y1W0R0-1X0Y2W1R0", # Oja's rule
    "generation_model": "volterra",
    "plasticity_coeffs_init": "random",
    "plasticity_model": "volterra",
    "plasticity_coeffs_init_scale": 1e-4,

    "regularization_type": "none",  # "l1", "l2", "none"
    "regularization_scale": 0,

    "fit_data": "neural",
    "neural_recording_sparsity": 1,
    "measurement_noise_scale": 0,

    # Restrictions on trainable plasticity parameters
    "trainable_coeffs": int(np.sum(coeff_mask)),
    "coeff_mask": coeff_mask.tolist(),

    "log_expdata": True,
    "data_dir": "../../../../03_data/01_original_data/",
    "log_dir": "../../../../03_data/02_training_data/",
}

cfg = OmegaConf.create(config)
#TODO cfg = validate_config(cfg)
key = jax.random.PRNGKey(cfg["expid"])


# -

def generate_experiments(key, cfg,
                         generation_coeff, generation_func,
                         global_teacher_init_params,
                         mode="train"):
    # Generate all experiments/trajectories
    if mode == "train":
        num_experiments = cfg.num_exp_train
        print(f"\nGenerating {num_experiments} trajectories")
    elif mode == "eval":
        num_experiments = cfg.num_exp_eval
        print(f"\nGenerating {num_experiments} trajectories")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    experiments = []
    # experiments_data = {}
    for exp_i in range(num_experiments):
        # Pick random number of sessions in this experiment given mean and std
        key, num_sessions = sample_truncated_normal(
            key, cfg["mean_num_sessions"], cfg["sd_num_sessions"]
        )
        exp = experiment.Experiment(exp_i, cfg,
                                    generation_coeff, generation_func,
                                    num_sessions,
                                    global_teacher_init_params)
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
key, plasticity_key, params_key = jax.random.split(key, 3)
#TODO add branching for experimental data
generation_coeff, generation_func = synapse.init_plasticity(
    plasticity_key, cfg, mode="generation_model"
)
global_teacher_init_params = model.initialize_parameters(
    params_key,
    cfg["num_hidden_pre"], cfg["num_hidden_post"],
    cfg["initial_params_scale"]
)
key, experiments = generate_experiments(
    key, cfg, generation_coeff, generation_func,
    global_teacher_init_params, mode="train",
)


# +
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
exp= 0
session = 0


inputs = experiments[exp].data["inputs"]   # (n_sess, n_steps, 1)
xs     = experiments[exp].data["xs"]       # (n_sess, n_steps, xdim)
ys     = experiments[exp].data["ys"]       # (n_sess, n_steps, ydim)

# n_sess = inputs.shape[0]

# # compute per-session order of step indices (ascending)
# order = jnp.argsort(jnp.squeeze(inputs, -1), axis=1)   # (n_sess, n_steps)

# # build row index for broadcasting:
# rows = jnp.arange(n_sess)[:, None]                      # (n_sess, 1)

# # apply same permutation to xs and ys:
# xs_sorted = xs[rows, order]   # (n_sess, n_steps, xdim)
# ys_sorted = ys[rows, order]   # (n_sess, n_steps, ydim)

xs_ax = ax[0].imshow(xs[session], aspect='auto',
                     cmap='viridis', interpolation='none')
ys_ax = ax[1].imshow(ys[session], aspect='auto',
                     cmap='viridis', interpolation='none')
# ws_ax = ax[2].imshow(experiments[exp].data['params'][0][0], aspect='auto',
                    #  cmap='viridis', interpolation='none')
fig.colorbar(xs_ax, ax=ax[0])
fig.colorbar(ys_ax, ax=ax[1])
# fig.colorbar(ws_ax, ax=ax[2])
ax[0].set_title('Presynaptic')
ax[1].set_title('Postsynaptic')
ax[0].set_ylabel('Time step')
ax[0].set_xlabel('Neuron')
ax[1].set_xlabel('Neuron')
ax[0].set_xlim(0-0.5, config["num_hidden_pre"])
ax[0].set_ylim(cfg["mean_steps_per_trial"], 0-0.5)
ax[1].set_xlim(0-0.5, config["num_hidden_post"])
ax[1].set_ylim(cfg["mean_steps_per_trial"], 0-0.5)
plt.show()

# -

# Print data
print(len(experiments[0].data))
yrange = 0
for exp in experiments:
    # print(f'{exp.exp_i=}')
    # print(f'{exp.data["inputs"].shape=}')
    # print(f'{exp.data["xs"].shape=}')
    # print(f'{exp.data["ys"].shape=}')
    # print(f'{jnp.mean(exp.input_params)=}')
    # print(f'{exp.steps_per_session=}')
    # print(f'{exp.params[0].shape=}')
    yrange += jnp.max(exp.data["ys"][0, -1]) - jnp.min(exp.data["ys"][0, -1])
print(f'{exp.data["ys"].shape=}')
print(f'{jnp.min(exp.data["xs"])=}, {jnp.max(exp.data["xs"])=}')
print(f'{jnp.min(exp.data["ys"][0, 0])=}, {jnp.max(exp.data["ys"][0, 0])=}')
print(f'{jnp.min(exp.data["ys"][0, -1])=}, {jnp.max(exp.data["ys"][0, -1])=}')
print(f'Average yrange: {yrange / len(experiments)}')

