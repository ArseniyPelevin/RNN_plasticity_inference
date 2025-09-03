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

import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import training
from matplotlib.lines import Line2D
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
    "num_exp_eval": 5,

    "input_firing_mean": 0,
    "input_firing_std": 1,  # Standard deviation of input firing rates
    "input_noise_std": 0,  # Standard deviation of noise added to presynaptic layer
    "synapse_learning_rate": 0.1,
    "learning_rate": 3e-3,

    "input_params_scale": 1,
    "init_params_scale": 0.1,  # float or 'Xavier'

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
    "expid": 17, # For saving results and seeding random

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

    "return_trajectories": False,
    "return_params_trajectory": False,

    "log_expdata": True,
    "log_interval": 10,
    "data_dir": "../../../../03_data/01_original_data/",
    "log_dir": "../../../../03_data/02_training_data/",
    "fig_dir": "../../../../05_figures/"
}
cfg = OmegaConf.create(config)
#TODO cfg = validate_config(cfg)
# -

def run_experiment():

    key = jax.random.PRNGKey(cfg["expid"])

    key, experiments = training.generate_data(key, cfg)

    time_start = time.time()
    key, plasticity_coeffs, plasticity_func, expdata, activation_trajs = training.train(
        key, cfg, experiments)
    train_time = time.time() - time_start

    key, expdata = training.evaluate_model(key, cfg,
                                      plasticity_coeffs, plasticity_func,
                                      expdata)

    training.save_results(cfg, expdata, train_time)

    return activation_trajs


# +
# Run Exp10-16
print("\nEXPERIMENT 10")
cfg.expid = 10
cfg.input_firing_std = 0.5
run_experiment()

print("\nEXPERIMENT 11")
cfg.expid = 11
cfg.input_firing_std = 0.1
run_experiment()

cfg.input_firing_std = 1

print("\nEXPERIMENT 12")
cfg.expid = 12
cfg.synapse_learning_rate = 0.5
run_experiment()

print("\nEXPERIMENT 13")
cfg.expid = 13
cfg.synapse_learning_rate = 1
run_experiment()

cfg.synapse_learning_rate = 0.1

print("\nEXPERIMENT 14")
cfg.expid = 14
cfg.init_params_scale = 0.05
run_experiment()

print("\nEXPERIMENT 15")
cfg.expid = 15
cfg.init_params_scale = 0.01
run_experiment()

cfg.init_params_scale = 0.1

print("\nEXPERIMENT 16")
cfg.expid = 16
run_experiment()

# -

def plot_coeff_trajectories(exp_id, params_table):
    """
    Plot a single experiment's loss (top) and coefficient trajectories (bottom).

    Args:
        exp_id (int): single experiment id
        params_table (dict): mapping exp_id -> dict of parameters for subplot title
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # single file path for the single experiment
    path = r"..\\..\\..\\..\\03_data\\02_training_data\\"
    fpath = os.path.join(path, f"exp_{exp_id}.csv")

    # highlight a few columns (if they exist)
    highlight = {"A_1100", "A_0210"}

    # top = Loss, bottom = coeff trajectories
    fig, axs = plt.subplots(2, 1, figsize=(12, 7))
    top_ax, coeff_ax = axs

    # --- read data ---
    df = pd.read_csv(fpath)

    # --- Top: loss subplot (backwards-compatible) ---
    epochs = df['epoch']

    if 'train_loss' in df.columns and 'test_loss' in df.columns:
        top_ax.plot(epochs, df['train_loss'], color='blue', label='train_loss')
        top_ax.plot(epochs, df['test_loss'], color='red', label='test_loss')
    elif 'loss' in df.columns:
        # older files had only 'loss' column — plot it as 'train_loss'
        top_ax.plot(epochs, df['loss'], color='blue', label='train_loss')

    top_ax.set_title("Loss")
    top_ax.legend(loc='upper right')
    top_ax.grid(True)
    top_ax.set_yscale('log')
    top_ax.set_ylabel('Loss (log scale)')

    # --- Bottom: coefficient trajectories ---
    # Heuristic: columns that end with a 4-digit suffix like "_abcd"
    # and we prefer those ending with '0'
    candidate_cols = [c for c in df.columns[:200]
                      if len(str(c).split('_')[-1]) == 4]
    data_cols = [c for c in candidate_cols
                 if str(c).split('_')[-1].endswith('0')]

    # Fallback: if nothing found, use all columns except loss/epoch
    if not data_cols:
        excluded = {'epoch', 'loss', 'train_loss', 'test_loss'}
        data_cols = [c for c in df.columns if c not in excluded]

    if not data_cols:
        coeff_ax.text(0.5, 0.5, "No coefficient columns found in CSV",
                      ha='center', va='center')
        plt.show()
        return

    # group by w-exponent (third digit of suffix) for coloring / styling
    groups = {}
    for c in data_cols:
        suffix = str(c).split('_')[-1]
        a, b, w, r = map(int, list(suffix))
        groups.setdefault(w, []).append(c)

    # deterministic ordering key
    def col_key(col):
        s = str(col).split('_')[-1]
        return (tuple(map(int, list(s))) if len(s) == 4 and s.isdigit()
                else (0, 0, 0, 0))

    # assign colors within each w-group
    color_map = {}
    for _wexp, cols in groups.items():
        cols_sorted = sorted(cols, key=col_key)
        n = len(cols_sorted)
        cmap = plt.get_cmap('Set1')
        colors = [cmap(0.5)] if n == 1 else [cmap(t)
                                             for t in np.linspace(0, 1, n)]
        for col, colcolor in zip(cols_sorted, colors, strict=False):
            color_map[col] = colcolor

    # pretty labels
    def pretty_label(col):
        suffix = str(col).split('_')[-1]
        a, b, c_, d = map(int, list(suffix))
        parts = []
        for exp, var in ((a, 'x'), (b, 'y'), (c_, 'w'), (d, 'r')):
            if exp == 0:
                continue
            if exp == 1:
                parts.append(var)
            else:
                parts.append(f"{var}^{{{exp}}}")
        return f"${''.join(parts)}$" if parts else col

    label_map = {c: pretty_label(c) for c in data_cols}
    linestyle_map = {0: '-', 1: '--', 2: ':'}

    x = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))
    ax = coeff_ax
    for c in data_cols:
        suffix = str(c).split('_')[-1]
        wexp = int(suffix[2])
        lw = 3 if c in highlight else 2
        ls = linestyle_map.get(wexp, '-')
        ax.plot(x, df[c], label=label_map.get(c, c), linewidth=lw, linestyle=ls,
                color=color_map.get(c))

    # title with params if available
    basename = os.path.basename(fpath)
    exp_num = int(basename.split('_')[1].split('.')[0])
    if exp_num is not None and exp_num in params_table:
        p = params_table[exp_num]
        param_str = ', '.join(f'{key}={value}' for key, value in p.items())
        ax.set_title(f"{basename[:-4]}: {param_str}", fontsize=12)
    else:
        ax.set_title(basename, fontsize=12)

    ax.grid(True)
    ax.set_xlabel('epoch')

    # single legend with pretty labels (ordered by data_cols)
    legend_handles = []
    legend_labels = []
    for c in data_cols:
        suffix = str(c).split('_')[-1]
        wexp = int(suffix[2])
        legend_handles.append(Line2D([0], [0], color=color_map.get(c),
                                     lw=(3 if c in highlight else 2),
                                     linestyle=linestyle_map.get(wexp, '-')))
        legend_labels.append(label_map.get(c, c))

    fig.subplots_adjust(bottom=0.2, hspace=0.35)

    # place legend below the figure (centered)
    fig.legend(legend_handles, legend_labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.04),
               ncol=9,
               fontsize=12,
               frameon=False,
               handlelength=2.0)

    plt.show()

    return fig


# Set parameters and run experiment
# # Reload synapse module:
importlib.reload(synapse)
importlib.reload(training)
importlib.reload(losses)
# reload synapse
cfg.expid = 34
cfg.num_exp_train = 25
cfg.num_hidden_pre = 10
cfg.num_hidden_post = 10
cfg.input_firing_std = 1
cfg.synapse_learning_rate = 1
cfg.init_params_scale = 0.1
activation_trajs = run_experiment()

# +
# Diagnose trajectories for NaN
print(len(activation_trajs)) # num epochs
print(len(activation_trajs[0])) # num experiments
print(len(activation_trajs[0][0])) # (x, y, output)
print(activation_trajs[0][0][0][0].shape)  # w.shape
print(activation_trajs[0][0][0][1].shape)  # b.shape
print(activation_trajs[0][0][1].shape)  # x.shape
print(activation_trajs[0][0][2].shape)  # y.shape
print(activation_trajs[0][0][3].shape)  # output.shape

def find_first_nan_y(activation_trajs):
    """
    Returns (epoch_idx, exp_idx, y_array(50,10)) for the first y containing NaN.
    If none found returns (None, None, None).
    """
    for e_idx, epoch in enumerate(activation_trajs):
        for ex_idx, trajs in enumerate(epoch):
            y = np.asarray(trajs[2]).squeeze()
            x = np.asarray(trajs[1]).squeeze()
            mask_rows = np.any(np.isnan(y), axis=1)
            rows = np.where(mask_rows)[0]
            if rows.size:
                return e_idx, ex_idx, int(rows[0]), x, y
    return None, None, None

# example usage:
epoch_i, exp_i, step_i, x_bad, y_bad = find_first_nan_y(activation_trajs)
print(epoch_i, exp_i, step_i)
# x_bad = activation_trajs[57][20][1][0]
# y_bad = activation_trajs[57][20][2][0]
# w = activation_trajs[57][20][0][0][0]

# +
# parameters table to include in subplot titles
params_table = {
     10: {'input_std': 0.5, 'synapse_lr': 0.1, 'init_w_std': 0.1,
          "N_in": 100, "N_out": 1000, "N_exp": 50},
     11: {'input_std': 0.1, 'synapse_lr': 0.1, 'init_w_std': 0.1,
          "N_in": 100, "N_out": 1000, "N_exp": 50},  # nan
     12: {'input_std': 1.0, 'synapse_lr': 0.5, 'init_w_std': 0.1,
          "N_in": 100, "N_out": 1000, "N_exp": 50},
     13: {'input_std': 1.0, 'synapse_lr': 1.0, 'init_w_std': 0.1,
          "N_in": 100, "N_out": 1000, "N_exp": 50},
     14: {'input_std': 1.0, 'synapse_lr': 0.1, 'init_w_std': 0.05,
          "N_in": 100, "N_out": 1000, "N_exp": 50},  # nan
     15: {'input_std': 1.0, 'synapse_lr': 0.1, 'init_w_std': 0.01,
          "N_in": 100, "N_out": 1000, "N_exp": 50},
     16: {'input_std': 1.0, 'synapse_lr': 0.1, 'init_w_std': 0.1,
          "N_in": 100, "N_out": 1000, "N_exp": 50},
     17: {'input_std': 1.0, 'synapse_lr': 0.5, 'init_w_std': 0.1,
         "N_in": 50, "N_out": 500, "N_exp": 25},
     18: {'input_std': 1.0, 'synapse_lr': 0.5, 'init_w_std': 0.1,
         "N_in": 10, "N_out": 10, "N_exp": 50},
     19: {'input_std': 1.0, 'synapse_lr': 0.5, 'init_w_std': 0.1,
          "N_in": 10, "N_out": 10, "N_exp": 25},
     20: {'input_std': 0.1, 'synapse_lr': 1.0, 'init_w_std': 'Xavier (1/10+10)',
          "N_in": 10, "N_out": 10, "N_exp": 25},
     21: {'input_std': 0.1, 'synapse_lr': 0.1, 'init_w_std': 'Xavier (1/10+10)',
          "N_in": 10, "N_out": 10, "N_exp": 25},
     22: {'input_std': 1, 'synapse_lr': 1, 'init_w_std': 'Xavier (1/10+10)',
          "N_in": 10, "N_out": 10, "N_exp": 25},
     23: {'input_std': 1, 'synapse_lr': 0.5, 'init_w_std': 'Xavier (1/10+10)',
         "N_in": 10, "N_out": 10, "N_exp": 25},
     24: {'input_std': 1, 'synapse_lr': 0.5, 'init_w_std': 'Xavier (1/100+100)',
         "N_in": 100, "N_out": 100, "N_exp": 25},
     25: {'input_std': 1, 'synapse_lr': 0.1, 'init_w_std': 'Xavier (1/10+10)',
         "N_in": 10, "N_out": 10, "N_exp": 25},
     26: {'input_std': 1, 'synapse_lr': 0.5, 'init_w_std': 'Xavier (1/10+10)',
         "N_in": 10, "N_out": 10, "N_exp": 25, "teacher/student init params": ""},
     27: {'input_std': 1, 'synapse_lr': 1, 'init_w_std': 'Xavier (1/10+10)',
         "N_in": 10, "N_out": 10, "N_exp": 25, "teacher/student init params": ""},
     28: {'input_std': 0.1, 'synapse_lr': 1, 'init_w_std': 'Xavier (1/10+10)',
         "N_in": 10, "N_out": 10, "N_exp": 25, "teacher/student init params": ""},
     29: {'input_std': 1, 'synapse_lr': 1, 'init_w_std': 'Xavier (1/100+100)',
         "N_in": 100, "N_out": 100, "N_exp": 25, "teacher/student init params": ""},
     30: {'input_std': 1, 'synapse_lr': 1, 'init_w_std': 0.01,
         "N_in": 100, "N_out": 100, "N_exp": 25, "teacher/student init params": ""},
     31: {'input_std': 1, 'synapse_lr': 1, 'init_w_std': 0.01,
         "N_in": 10, "N_out": 10, "N_exp": 25},
     32: {'input_std': 1, 'synapse_lr': 1, 'init_w_std': 0.001,
         "N_in": 10, "N_out": 10, "N_exp": 25},
     33: {'input_std': 1, 'synapse_lr': 1, 'init_w_std': 0.0001,
         "N_in": 10, "N_out": 10, "N_exp": 25},
     34: {'input_std': 1, 'synapse_lr': 1, 'init_w_std': 0.1,
         "N_in": 10, "N_out": 10, "N_exp": 25},
}

fig = plot_coeff_trajectories(cfg.expid, params_table)
fig.savefig(cfg.fig_dir + f"Exp{cfg.expid} coeff trajectories.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
# -

key = jax.random.PRNGKey(cfg["expid"])
key, experiments = training.generate_data(key, cfg)

# +
# Plot xs and ys
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
exp= 0
session = 0

inputs = experiments[exp].data["inputs"]   # (n_sess, n_steps, 1)
xs     = experiments[exp].data["xs"]       # (n_sess, n_steps, xdim)
ys     = experiments[exp].data["ys"]       # (n_sess, n_steps, ydim)

# # Sort by input (if discrete input classes)
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
ax[0].set_xlim(0-0.5, cfg["num_hidden_pre"])
ax[0].set_ylim(cfg["mean_steps_per_trial"], 0-0.5)
ax[1].set_xlim(0-0.5, cfg["num_hidden_post"])
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

