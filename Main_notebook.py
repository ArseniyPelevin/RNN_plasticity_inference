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

import itertools
import os
import time
from importlib import reload

import experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import model
import numpy as np
import pandas as pd
import training
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from omegaconf import OmegaConf

# +
# coeff_mask = np.zeros((3, 3, 3, 3))
# coeff_mask[0:2, 0, 0, 0:2] = 1
coeff_mask = np.ones((3, 3, 3, 3))
coeff_mask[:, :, :, 1:] = 0  # Zero out reward coefficients

config = {
    "expid": 17, # For saving results and seeding random
    "use_experimental_data": False,
    "fit_data": "neural",  # ["behavioral", "neural"]

# Experiment design
    "num_exp_train": 25,  # Number of experiments/trajectories/animals
    "num_exp_test": 5,
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

# Network architecture
    "num_inputs": 6,  # Number of input classes (num_epochs * 4 for random normal)
    "num_hidden_pre": 50,  # x, presynaptic neurons for plasticity layer
    "num_hidden_post": 50,  # y, postsynaptic neurons for plasticity layer
    "num_outputs": 1,  # m, binary decision (licking/not licking at this time step)
    "recurrent": True,  # Whether to include recurrent connections
    "plasticity_layers": ["recurrent"],  # ["feedforward", "recurrent"]
    "postsynaptic_input_sparsity": 1,  # Fraction of posts. neurons receiving FF input,
        # only effective if recurrent connections are present, otherwise 1
    "feedforward_sparsity": 0.2,  # Fraction of nonzero weights in feedforward layer,
        # of all postsynaptic neurons receiving FF input (postsynaptic_input_sparsity),
        # all presynaptic neurons are guaranteed to have some output
    "recurrent_sparsity": 1,  # Fraction of nonzero weights in recurrent layer,
        # all neurons receive some input (FF or rec, not counting self-connections)
    "neural_recording_sparsity": 1,

# Network dynamics
    "input_params_scale": 1,
    "presynaptic_firing_mean": 0,
    "presynaptic_firing_std": 1,  # Input (before presynaptic) firing rates
    "presynaptic_noise_std": 0,  #0.05 # Noise added to presynaptic layer
    "feedforward_input_scale": 1,  # Scale of feedforward weights,
        # only if no feedforward plasticity
    "recurrent_input_scale": 1,  # Scale of recurrent weights,
        # only if no recurrent plasticity
    "init_params_scale": 0.01,  # float or 'Xavier'
    "reward_scale": 0,
    "synaptic_weight_threshold": 6,  # Weights are normally in the range [-4, 4]
    "synapse_learning_rate": 1,
    "measurement_noise_scale": 0,

# Plasticity
    "generation_plasticity": "1X1Y1W0R0-1X0Y2W1R0", # Oja's rule
    "generation_model": "volterra",
    "plasticity_coeffs_init": "random",
    "plasticity_model": "volterra",
    "plasticity_coeffs_init_scale": 1e-4,
    # Restrictions on trainable plasticity parameters
    "trainable_coeffs": int(np.sum(coeff_mask)),
    "coeff_mask": coeff_mask.tolist(),

# Training
    "num_epochs": 250,
    "learning_rate": 3e-3,
    "regularization_type": "none",  # "l1", "l2", "none"
    "regularization_scale": 0,

# Logging
    "log_expdata": True,
    "log_interval": 10,
    "data_dir": "../../../../03_data/01_original_data/",
    "log_dir": "../../../../03_data/02_training_data/",
    "fig_dir": "../../../../05_figures/",

    "_return_params_trajec": False,  # For debugging
}
cfg = OmegaConf.create(config)
#TODO cfg = validate_config(cfg)
# -

def run_experiment():

    key = jax.random.PRNGKey(cfg["expid"])
    # Pass subkeys, so that adding more experiments doesn't affect earlier ones
    train_exp_key, test_exp_key, train_key, eval_key = jax.random.split(key, 4)

    experiments = training.generate_data(train_exp_key, cfg, mode='train')
    test_experiments = training.generate_data(test_exp_key, cfg, mode='test')

    time_start = time.time()
    plasticity_coeffs, plasticity_func, expdata, _activation_trajs = (
        training.train(train_key, cfg, experiments, test_experiments))
    train_time = time.time() - time_start

    expdata = training.evaluate_model(eval_key, cfg,
                                      test_experiments,
                                      plasticity_coeffs, plasticity_func,
                                      expdata)
    training.save_results(cfg, expdata, train_time)
    return _activation_trajs


# +
# Run Exp10-16
print("\nEXPERIMENT 10")
cfg.expid = 10
cfg.presynaptic_firing_std = 0.5
run_experiment()

print("\nEXPERIMENT 11")
cfg.expid = 11
cfg.presynaptic_firing_std = 0.1
run_experiment()

cfg.presynaptic_firing_std = 1

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

def plot_coeff_trajectories(exp_id, params_table, use_all_81=False):
    """
    Plot a single experiment's loss (top) and coefficient trajectories (bottom).

    This function is 100% ChatGPT 5 generated

    Args:
        exp_id (int): single experiment id
        params_table (dict): mapping exp_id -> dict of parameters for subplot title
    """

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
    x_epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))

    if all(col in df.columns for col in ['train_loss_mean','train_loss_std',
                                         'test_loss_mean','test_loss_std']):
        top_ax.plot(x_epochs, df['train_loss_mean'], color='blue', label='train_loss')
        top_ax.fill_between(x_epochs,
                            df['train_loss_mean'] - df['train_loss_std'],
                            df['train_loss_mean'] + df['train_loss_std'],
                            color='blue', alpha=0.2)
        top_ax.plot(x_epochs, df['test_loss_mean'], color='red', label='test_loss')
        top_ax.fill_between(x_epochs,
                            df['test_loss_mean'] - df['test_loss_std'],
                            df['test_loss_mean'] + df['test_loss_std'],
                            color='red', alpha=0.2)
    elif 'train_loss' in df.columns and 'test_loss' in df.columns:
        top_ax.plot(x_epochs, df['train_loss'], color='blue', label='train_loss')
        top_ax.plot(x_epochs, df['test_loss'], color='red', label='test_loss')
    elif 'loss' in df.columns:
        top_ax.plot(x_epochs, df['loss'], color='blue', label='train_loss')

    top_ax.set_title("Loss")
    top_ax.legend(loc='upper right')
    top_ax.grid(True)
    top_ax.set_yscale('log')
    top_ax.set_ylabel('Loss (log scale)')

    # --- Bottom: coefficient trajectories ---
    # find candidate columns like "A_abcd"
    candidate_cols = [c for c in df.columns if len(str(c).split('_')[-1]) == 4
                      and str(c).split('_')[-1].isdigit()]

    if use_all_81:
        data_cols = candidate_cols  # use all 81 A_xxxx columns
    else:
        # old behaviour: only those ending with '0' (27 columns)
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
    parsed = {}
    for c in data_cols:
        suffix = str(c).split('_')[-1]
        a, b, w, r = map(int, list(suffix))
        groups.setdefault(w, []).append(c)
        parsed[c] = (a, b, w, r)

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

    # pretty labels (now include r as well)
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
    linestyle_map = {0: '-', 1: '--', 2: ':'}  # W^0,W^1,W^2

    x = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))
    ax = coeff_ax

    # jitter in axis units: small horizontal offset for R groups
    jitter = 1.0
    x_vals = np.asarray(x)

    # plot lines + markers according to R
    for c in sorted(data_cols, key=col_key):
        a, b, wexp, rexp = parsed[c]
        lw = 3 if c in highlight else 2
        ls = linestyle_map.get(wexp, '-')
        color = color_map.get(c, 'k')
        # shift both line and markers by rexp * jitter (in axis units)
        x_plot = x_vals + (rexp * jitter)
        ax.plot(x_plot, df[c], label=label_map.get(c, c),
                linewidth=lw, linestyle=ls, color=color)
        # overlay markers for R=1 and R=2 (same shift)
        if rexp == 1:
            ax.plot(x_plot, df[c], linestyle='None', marker='o', markersize=4,
                    markerfacecolor=color, markeredgecolor=color)
        elif rexp == 2:
            ax.plot(x_plot, df[c], linestyle='None', marker='o', markersize=4,
                    markerfacecolor='none', markeredgecolor=color)

    # title with params if available
    basename = os.path.basename(fpath)
    exp_num = exp_id
    if exp_num is not None and exp_num in params_table:
        p = params_table[exp_num]
        param_str = ', '.join(f'{key}={value}' for key, value in p.items())
        ax.set_title(f"{basename[:-4]}: {param_str}", fontsize=12)
    else:
        ax.set_title(basename, fontsize=12)

    ax.grid(True)
    ax.set_xlabel('epoch')

    # build legend handles for each R block separately
    # (preserve same base 27 order)
    # create ordered list of 27 unique (a,b,w) combos
    base_keys = []
    seen = set()
    for c in sorted(data_cols, key=col_key):
        a,b,w,r = parsed[c]
        key3 = (a,b,w)
        if key3 not in seen:
            seen.add(key3)
            base_keys.append(key3)

    # helper to make a proxy handle (include marker for R1/R2)
    def proxy_handle(col, rexp):
        a, b, wexp, _ = parsed[col]
        color = color_map.get(col, 'k')
        ls = linestyle_map.get(wexp, '-')
        lw = 3 if col in highlight else 2
        if rexp == 0:
            return Line2D([0], [0], color=color, lw=lw, linestyle=ls)
        if rexp == 1:
            return Line2D([0], [0], color=color, lw=lw, linestyle=ls,
                          marker='o', markerfacecolor=color, markersize=6)
        return Line2D([0], [0], color=color, lw=lw, linestyle=ls,
                      marker='o', markerfacecolor='none', markersize=6)

    # now build handles/labels per R
    handles_by_r = {0: [], 1: [], 2: []}
    labels_by_r = {0: [], 1: [], 2: []}
    for rexp in (0,1,2):
        for key3 in base_keys:
            # find column with this (a,b,w) and r=rexp
            target = next((cc for cc in data_cols
                           if parsed[cc][:3] == key3
                           and parsed[cc][3] == rexp), None)
            if target is None:
                h = Line2D([0],[0], color='none')
                lbl = ''
            else:
                h = proxy_handle(target, rexp)
                lbl = label_map.get(target, target)
            handles_by_r[rexp].append(h)
            labels_by_r[rexp].append(lbl)

    if use_all_81:
        # --- Build legend with columns = XY combos (9 columns)
        # and rows = (w,r) combos (9 rows)
        # Desired per-column order for a given (x,y):
        # (w,r) = (0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)
        xy_keys = []
        seen_xy = set()
        for c in sorted(data_cols, key=col_key):
            a, b, w, r = parsed[c]
            key_ab = (a, b)
            if key_ab not in seen_xy:
                seen_xy.add(key_ab)
                xy_keys.append(key_ab)

        # if we don't have exactly 9 xy keys,
        # fall back to prior base_keys grouping
        if len(xy_keys) != 9:
            # fallback: group by (a,b,w) order
            # and derive 9 XY keys by selecting unique (a,b)
            xy_keys = []
            seen_xy = set()
            for c in sorted(data_cols, key=col_key):
                a, b, w, r = parsed[c]
                key_ab = (a, b)
                if key_ab not in seen_xy:
                    seen_xy.add(key_ab)
                    xy_keys.append(key_ab)
            # trim or pad to 9 if needed
            xy_keys = (xy_keys + [xy_keys[-1]]*9)[:9]

        # build display-order arrays (row-major layout: rows=9, cols=9)
        ncol = 9
        nrows = 9
        total = ncol * nrows
        display_handles = [None] * total
        display_labels = [None] * total

        # desired (w,r) sequence per column
        wr_seq = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)]

        for col_idx, (a, b) in enumerate(xy_keys):
            for row_idx, (wexp, rexp) in enumerate(wr_seq):
                display_index = row_idx * ncol + col_idx  # row-major position
                # find the column that matches (a,b,wexp,rexp)
                target = next((cc for cc in data_cols
                               if parsed[cc][:2] == (a, b)
                               and parsed[cc][2] == wexp
                               and parsed[cc][3] == rexp), None)
                if target is None:
                    display_handles[display_index] = Line2D(
                        [0], [0], color='none')
                    display_labels[display_index] = ''
                else:
                    display_handles[display_index] = proxy_handle(target, rexp)
                    display_labels[display_index] = label_map.get(target, target)

        # Matplotlib fills legend entries column-major;
        # reorder so the final displayed grid (row-major)
        # matches our display_handles list.
        reordered_handles = [None] * total
        reordered_labels = [None] * total
        for i in range(total):
            row = i % nrows
            col = i // nrows
            display_index = row * ncol + col
            reordered_handles[i] = display_handles[display_index]
            reordered_labels[i] = display_labels[display_index]

        # expand bottom space to fit legend and avoid overlap
        fig.subplots_adjust(bottom=0.46, hspace=0.70, top=0.88)

        # place single legend below the figure (centered) with ncol columns
        fig.legend(reordered_handles, reordered_labels,
                   loc='lower center',
                   bbox_to_anchor=(0.5, -0.02),
                   bbox_transform=fig.transFigure,
                   ncol=ncol,
                   fontsize=10,
                   frameon=False,
                   handlelength=2.0,
                   markerscale=1.0,
                   labelspacing=0.4,
                   columnspacing=1.0,
                   handletextpad=0.5,
                   handleheight=1.0,
                   borderaxespad=0.5)

    else:
        # older 27-parameter mode: keep legend tight and close to axes (no huge blank)
        # build handles in the original 27-order (base_keys with r=0)
        legend_handles = []
        legend_labels = []
        for key3 in base_keys:
            # prefer r==0 column
            target = next((cc for cc in data_cols
                           if parsed[cc][:3] == key3
                           and parsed[cc][3] == 0), None)
            if target is None:
                # fallback to any representative
                target = next((cc for cc in data_cols
                               if parsed[cc][:3] == key3), None)
            if target is None:
                legend_handles.append(Line2D([0], [0], color='none'))
                legend_labels.append('')
            else:
                legend_handles.append(proxy_handle(target, 0))
                legend_labels.append(label_map.get(target, target))
        fig.subplots_adjust(bottom=0.18, hspace=0.50, top=0.92)
        ncol = 9
        fig.legend(legend_handles, legend_labels,
                   loc='lower center', bbox_to_anchor=(0.5, -0.04),
                   bbox_transform=fig.transFigure,
                   ncol=ncol,
                   fontsize=11, frameon=False, handlelength=2.4,
                   markerscale=1.0, labelspacing=0.30, columnspacing=1.0,
                   handletextpad=0.5,
                   handleheight=1.0, borderaxespad=0.5)

    # ensure x-axis ticks are sparse (every 5th epoch) but markers remain at each epoch
    # choose ticks based on epoch *values* (even if epoch spacing isn't uniform)
    if len(x_vals) > 1:
        epoch_step = np.median(np.diff(x_vals))
        desired_interval = 5 * epoch_step
        # select epoch values that are multiples of desired_interval (within tolerance)
        tol = max(1e-8, desired_interval * 0.01)
        mask = np.isclose(((x_vals - x_vals[0]) % desired_interval), 0, atol=tol)
        tick_positions = x_vals[mask]
        # fallback if mask selects too few points: use index-based every-5th
        if len(tick_positions) < 2:
            idxs = np.arange(0, len(x_vals), 5)
            if idxs[-1] != len(x_vals)-1:
                idxs = np.concatenate([idxs, [len(x_vals)-1]])
            tick_positions = x_vals[idxs]
    else:
        tick_positions = x_vals

    coeff_ax.set_xticks(tick_positions)
    coeff_ax.set_xticklabels([str(int(v)) if float(v).is_integer() else str(v)
                              for v in tick_positions])

    plt.show()

    return fig


# +
# Explore space of input-output layer sizes

cfg.num_exp_train = 25
cfg.presynaptic_noise_std = 0
cfg.presynaptic_firing_std = 1
cfg.synapse_learning_rate = 1
cfg.init_params_scale = 0.01

for i, (N_in, N_out) in enumerate(list(itertools.product([10, 50, 100, 500, 1000],
                                      [10, 50, 100, 500, 1000]))):
    cfg.num_hidden_pre = N_in
    cfg.num_hidden_post = N_out
    cfg.expid = 50 + i
    run_experiment()
    params_dict = {cfg.expid: {"N_in": N_in, "N_out": N_out}}
    fig = plot_coeff_trajectories(cfg.expid, params_dict)
    fig.savefig(cfg.fig_dir + f"Exp{cfg.expid} coeff trajectories.png",
            dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted Exp{cfg.expid} with N_in={N_in}, N_out={N_out}")

# +
# Diagnose trajectories for NaN

_activation_trajs = run_experiment()
print(len(_activation_trajs)) # num epochs
print(len(_activation_trajs[0])) # num experiments
print(len(_activation_trajs[0][0])) # (x, y, output)
print(_activation_trajs[0][0]['params'][0].shape)  # w.shape
print(_activation_trajs[0][0]['params'][1].shape)  # b.shape
print(_activation_trajs[0][0]['xs'].shape)  # x.shape
print(_activation_trajs[0][0]['ys'].shape)  # y.shape
print(_activation_trajs[0][0]['outputs'].shape)  # output.shape

def find_first_nan_y(_activation_trajs):
    """
    Returns (epoch_idx, exp_idx, y_array(50,10)) for the first y containing NaN.
    If none found returns (None, None, None).
    """
    for e_idx, epoch in enumerate(_activation_trajs):
        for ex_idx, trajs in enumerate(epoch):
            ys = np.asarray(trajs['ys'])
            xs = np.asarray(trajs['xs'])
            for sess_idx in range(ys.shape[0]):
                x = xs[sess_idx]
                y = ys[sess_idx]
                mask_rows = np.any(np.isnan(y), axis=1)
                rows = np.where(mask_rows)[0]
                if rows.size:
                    return e_idx, ex_idx, sess_idx, int(rows[0]), x, y
    return None, None, None

# example usage:
epoch_i, exp_i, sess_i, step_i, x_bad, y_bad = find_first_nan_y(_activation_trajs)
print(f'{epoch_i=}, {exp_i=}, {sess_i=}, {step_i=}')
# x_bad = _activation_trajs[57][20][1][0]
# y_bad = _activation_trajs[57][20][2][0]
# w = _activation_trajs[57][20][0][0][0]

# +
# Set parameters and run experiment
training = reload(training)
experiment = reload(experiment)
model = reload(model)

# parameters table to include in subplot titles
# import json
# with open("feedforward_experiments_config_table.json", "r") as f:
#     feedforward_experiments_config_table = json.load(f)

recurrent_experiments_config_table = {
     1: {'plasticity': "recurrent", "N_in": 50, "N_out": 50,
         "\ninp_spar": 1, "FF_spar": 0.2, "rec_spar": 1, "FF_scale": 1,
         },
}

cfg.expid = 1
cfg.num_hidden_pre = 50
cfg.num_hidden_post = 50
cfg.recurrent = True
cfg.plasticity_layers = ["recurrent"]
cfg.postsynaptic_input_sparsity = 1
cfg.feedforward_sparsity = 0.2
cfg.recurrent_sparsity = 1
cfg.feedforward_input_scale = 1
cfg.recurrent_input_scale = 1

cfg.generation_plasticity = "1X1Y1W0R0-1X0Y2W1R0"  # Oja's

# _activation_trajs = run_experiment()

fig = plot_coeff_trajectories(cfg.expid, recurrent_experiments_config_table,
                              use_all_81=False)
fig.savefig(cfg.fig_dir + f"RNN_Exp{cfg.expid} coeff trajectories.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
# +
# Plot xs and ys, optionally evolution of weights

# _activation_trajs, _model_activations, _null_activations = run_experiment()

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
epoch = 8
exp = 3
session = 0

# inputs = experiments[exp].data["inputs"]   # (n_sess, n_steps, 1)
# xs = experiments[exp].data["xs"][0]  # (n_sess, n_steps, xdim)
# ys = experiments[exp].data["ys"][0]  # (n_sess, n_steps, ydim)
xs = _activation_trajs[epoch][exp]['xs'][session]
ys = _activation_trajs[epoch][exp]['ys'][session]
w = _activation_trajs[epoch][exp]['params'][0][session]
# xs = x_bad
# ys = y_bad

# # Sort by input (if discrete input classes)
# n_sess = inputs.shape[0]

# # compute per-session order of step indices (ascending)
# order = jnp.argsort(jnp.squeeze(inputs, -1), axis=1)   # (n_sess, n_steps)

# # build row index for broadcasting:
# rows = jnp.arange(n_sess)[:, None]                      # (n_sess, 1)

# # apply same permutation to xs and ys:
# xs_sorted = xs[rows, order]   # (n_sess, n_steps, xdim)
# ys_sorted = ys[rows, order]   # (n_sess, n_steps, ydim)

vmin, vmax = -2, 30
print(jnp.min(w), jnp.max(w))
# vmin, vmax = jnp.min(w), jnp.max(w)
fig = plt.figure(figsize=(12, 12))
# gs = fig.add_gridspec(1, 1)
# two rows: top - x and y, bottom - w evolution
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.2)

# Top: plot x and y
top_gs = gs[0].subgridspec(1, 2, wspace=0.25)
ax_xs = fig.add_subplot(top_gs[0, 0])
ax_ys = fig.add_subplot(top_gs[0, 1])

im_xs = ax_xs.imshow(xs, aspect='auto', cmap='viridis', interpolation='none')
im_ys = ax_ys.imshow(ys, aspect='auto', cmap='viridis', interpolation='none')

# colorbars for top axes (narrow to the right)
for ax, im in [(ax_xs, im_xs), (ax_ys, im_ys)]:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(im, cax=cax)

ax_xs.set_title('Presynaptic')
ax_ys.set_title('Postsynaptic')
ax_xs.set_ylabel('Time step')
ax_xs.set_xlabel('Neuron')
ax_ys.set_xlabel('Neuron')

ax_xs.set_xlim(-0.5, cfg["num_hidden_pre"])
ax_xs.set_ylim(cfg["mean_steps_per_trial"], -0.5)
ax_ys.set_xlim(-0.5, cfg["num_hidden_post"])
ax_ys.set_ylim(cfg["mean_steps_per_trial"], -0.5)

# Bottom: w evolution
num_rows = 4
bot_gs = gs[1].subgridspec(num_rows, 5, hspace=0.35, wspace=0.25)
axs = []
for r in range(num_rows):
    for c in range(5):
        axs.append(fig.add_subplot(bot_gs[r, c]))

for idx, step in enumerate(range(20, 40)):
    ax = axs[idx]
    im = ax.imshow(w[step], aspect='equal', cmap='viridis',
                   interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title(f'step {step}')
    ax.set_xticks([])
    ax.set_yticks([])

    # small colorbar to the right of each image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    fig.colorbar(im, cax=cax)

# remove any unused axes (if any)
for j in range(len(range(20, 40)), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()

# +
# Print data
key = jax.random.PRNGKey(cfg["expid"])
key, experiments = training.generate_data(key, cfg)

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

# -

# Plot weights distribution
flat = np.concatenate([exp['params'][0].ravel() for epoch in _activation_trajs
                       for exp in epoch])
print(flat.shape)
flat = flat[flat < 50]
fig, ax = plt.subplots()
ax.hist(flat, bins=100)
ax.set_xlabel("Synaptic weight")
ax.set_xlim(-5, 5)
plt.show()
