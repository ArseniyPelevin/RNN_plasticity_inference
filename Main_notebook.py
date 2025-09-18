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

import jax
import jax.numpy as jnp
import main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import training
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


# +
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

    # title with parameters if available
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

cfg = main.create_config()

# +
# Set parameters and run experiment
# training = reload(training)
# experiment = reload(experiment)
# model = reload(model)

# parameters table to include in subplot titles
# import json
# with open("feedforward_experiments_config_table.json", "r") as f:
#     feedforward_experiments_config_table = json.load(f)

recurrent_experiments_config_table = {
     1: {'plasticity': "recurrent", "N_in": 50, "N_out": 50,
         "\ninp_spar": 1, "FF_spar": 0.2, "rec_spar": 1, "FF_scale": 1,
         },
     11: {'plasticity': "recurrent", "N_in": 50, "N_out": 50,
          "\ninp_spar": 0.2, "FF_spar": 0.2, "rec_spar": 1, "FF_scale": 1,
          },
     16: {'recurrent': False, 'plasticity': "feedforward", "N_in": 10, "N_out": 10,
          "\ninp_spar": 1, "FF_spar": 1, "rec_spar": 1,
          },
     17: {'recurrent': True, 'plasticity': "feedforward", "N_in": 10, "N_out": 10,
          "\ninp_spar": 1, "FF_spar": 1, "rec_spar": 1,
          },
     18: {'recurrent': True, 'plasticity': "feedforward+recurrent",
          "N_in": 10, "N_out": 10,
          "\ninp_spar": 1, "FF_spar": 1, "rec_spar": 1,
          },
     19: {'recurrent': True, 'plasticity': "recurrent", "N_in": 10, "N_out": 10,
          "\ninp_spar": 1, "FF_spar": 1, "rec_spar": 1,
          },
     57: {'recurrent': False, 'plasticity': "feedforward", "N_in": 50, "N_out": 50,
          "\nFF_spar_gen": 1, "FF_spar_train": 1, "scale w by sqrt(N_in_i)": True
          },
     58: {'recurrent': False, 'plasticity': "feedforward", "N_in": 50, "N_out": 50,
          "\nFF_spar_gen": 1, "FF_spar_train": 1, "scale w by sqrt(N_in_i)": False
          },
     59: {'recurrent': False, 'plasticity': "feedforward", "N_in": 50, "N_out": 50,
          "\nFF_spar_gen": 1, "FF_spar_train": 1, "scale w by N_in_i": True
          },
     60: {'recurrent': True, 'plasticity': "ff", "N_in": 50, "N_out": 50},
     61: {'recurrent': True, 'plasticity': "ff, rec", "N_in": 50, "N_out": 50},
     62: {'recurrent': True, 'plasticity': "ff, rec", "N_in": 50, "N_out": 50,
          "\ninp_spar_gen": 0.5, "FF_spar_gen": 0.5, "rec_spar_gen": 0.5,
          "\ninp_spar_train": 1, "FF_spar_train": 1, "rec_spar_train": 1,
          "init_spar_gen": "{'ff': 1, 'rec': 1}",
          },
}

cfg = main.create_config()

cfg.recurrent = True
cfg.trainable_init_weights = ["w_rec", "w_ff"] #["w_ff"] #["w_rec", "w_ff"]
cfg.plasticity_layers = ["ff", "rec"]
cfg.postsynaptic_input_sparsity_generation = 1
cfg.postsynaptic_input_sparsity_training = 1
cfg.feedforward_sparsity_generation = 1
cfg.feedforward_sparsity_training = 1
cfg.recurrent_sparsity_generation = 1
cfg.recurrent_sparsity_training = 1

cfg.presynaptic_firing_mean = 0

# cfg.init_weights_sparsity_generation = {'ff': 0.5, 'rec': 0.5}
# cfg.init_weights_mean_generation = {'ff': 2, 'rec': -1, 'out': 0}
# cfg.init_weights_std_generation = {'ff': 0.01, 'rec': 1, 'out': 0}
# cfg.init_weights_std_training = {'ff': 1, 'rec': 1, 'out': 0}
cfg.init_weights_sparsity_generation = {'ff': 1, 'rec': 1}
cfg.init_weights_mean_generation = {'ff': 0, 'rec': 0, 'out': 0}
cfg.init_weights_std_generation = {'ff': 0.01, 'rec': 0.01, 'out': 0}
cfg.init_weights_std_training = {'ff': 0.01, 'rec': 0.01, 'out': 0}

# Exp57 = scaling by number of inputs, ff sparsity = 1
cfg.expid = 171
cfg.num_hidden_pre = 100
cfg.num_hidden_post = 100
cfg.mean_steps_per_trial = 50

cfg.num_epochs = 300

cfg.generation_plasticity = "1X1Y1W0R0-1X0Y2W1R0"  # Oja's

_activation_trajs = main.run_experiment(cfg)

fig = plot_coeff_trajectories(cfg.expid, recurrent_experiments_config_table,
                              use_all_81=False)
fig.savefig(cfg.fig_dir + f"RNN_Exp{cfg.expid} coeff trajectories.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
# +
# Plot histograms of losses and R2 values for all experiments

fig, ax = plt.subplots(2, 2, figsize=(10, 10), layout='tight')
ax = ax.flatten()
for i, metric in enumerate(['loss', 'MSE', 'r2_y', 'r2_w']):
    dF = losses_and_r2['F'][f'{metric}_all'].ravel()
    dN = losses_and_r2['N'][f'{metric}_all'].ravel()

    lo = np.nanmin([dF.min(), dN.min()])
    hi = np.nanmax([dF.max(), dN.max()])
    edges = np.linspace(lo, hi, 41)      # 40 bins -> 41 edges

    ax[i].hist(dF, bins=edges, alpha=0.5, color='blue', label='F')
    ax[i].hist(dN, bins=edges, alpha=0.5, color='red',  label='N')

    ax[i].set_title(metric)
    ax[i].legend()
plt.show()


# +
# Exp63-170
def run(cfg):
    _activation_trajs = main.run_experiment(cfg)

    params_table = {cfg.expid: {
        'trainable': str(cfg.trainable_init_weights),
        'inp_spar': cfg.postsynaptic_input_sparsity_generation,
        'ff_spar': cfg.feedforward_sparsity_generation,
        'rec_spar': cfg.recurrent_sparsity_generation,
        'init_spar_gen': cfg.init_weights_sparsity_generation,
        }}

    fig = plot_coeff_trajectories(cfg.expid, params_table,
                                use_all_81=False)
    fig.savefig(cfg.fig_dir + f"RNN_Exp{cfg.expid} coeff trajectories.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

cfg.recurrent = True
cfg.plasticity_layers = ["ff", "rec"]
cfg.init_weights_std_training = {'ff': 0.01, 'rec': 0.01, 'out': 0}
last_expid = 62
cfg.num_hidden_pre = 100
cfg.num_hidden_post = 100
cfg.mean_steps_per_trial = 50
cfg.num_epochs = 300
i = 1

for trainable in [[], ["w_rec"], ["w_rec", "w_ff"]]:
    cfg.trainable_init_weights = trainable
    for input_spars in [0.3, 0.6, 1]:
        for ff_spars in [0.3, 0.6, 1]:
            for rec_spars in [0.3, 0.6, 1]:
                cfg.postsynaptic_input_sparsity_generation = input_spars
                cfg.postsynaptic_input_sparsity_training = input_spars
                cfg.feedforward_sparsity_generation = ff_spars
                cfg.feedforward_sparsity_training = ff_spars
                cfg.recurrent_sparsity_generation = rec_spars
                cfg.recurrent_sparsity_training = rec_spars

                cfg.expid = last_expid + i

                run(cfg)
                i += 1

    cfg.postsynaptic_input_sparsity_generation = 1
    cfg.postsynaptic_input_sparsity_training = 1
    cfg.feedforward_sparsity_generation = 1
    cfg.feedforward_sparsity_training = 1
    cfg.recurrent_sparsity_generation = 1
    cfg.recurrent_sparsity_training = 1

    for init_spar_ff in [0.3, 0.6, 1]:
        for init_spar_rec in [0.3, 0.6, 1]:
            cfg.init_weights_sparsity_generation = {'ff': init_spar_ff,
                                                    'rec': init_spar_rec}
            cfg.expid = last_expid + i
            run(cfg)
            i += 1

# cfg.init_weights_sparsity_generation = {'ff': 0.5, 'rec': 0.5}
# cfg.init_weights_mean_generation = {'ff': 2, 'rec': -1, 'out': 0}
# cfg.init_weights_std_generation = {'ff': 0.01, 'rec': 1, 'out': 0}
# cfg.init_weights_std_training = {'ff': 1, 'rec': 1, 'out': 0}




# +
# Run Exp10-16
print("\nEXPERIMENT 10")
cfg.expid = 10
cfg.presynaptic_firing_std = 0.5
main.run_experiment()

print("\nEXPERIMENT 11")
cfg.expid = 11
cfg.presynaptic_firing_std = 0.1
main.run_experiment()

cfg.presynaptic_firing_std = 1

print("\nEXPERIMENT 12")
cfg.expid = 12
cfg.synapse_learning_rate = 0.5
main.run_experiment()

print("\nEXPERIMENT 13")
cfg.expid = 13
cfg.synapse_learning_rate = 1
main.run_experiment()

cfg.synapse_learning_rate = 0.1

print("\nEXPERIMENT 14")
cfg.expid = 14
cfg.init_weights_std = 0.05
main.run_experiment()

print("\nEXPERIMENT 15")
cfg.expid = 15
cfg.init_weights_std = 0.01
main.run_experiment()

cfg.init_weights_std = 0.1

print("\nEXPERIMENT 16")
cfg.expid = 16
main.run_experiment()


# +
# Explore space of input-output layer sizes

cfg.num_exp_train = 25
cfg.presynaptic_noise_std = 0
cfg.presynaptic_firing_std = 1
cfg.synapse_learning_rate = 1
cfg.init_weights_std = 0.01

for i, (N_in, N_out) in enumerate(list(itertools.product([10, 50, 100, 500, 1000],
                                      [10, 50, 100, 500, 1000]))):
    cfg.num_hidden_pre = N_in
    cfg.num_hidden_post = N_out
    cfg.expid = 50 + i
    main.run_experiment()
    params_dict = {cfg.expid: {"N_in": N_in, "N_out": N_out}}
    fig = plot_coeff_trajectories(cfg.expid, params_dict)
    fig.savefig(cfg.fig_dir + f"Exp{cfg.expid} coeff trajectories.png",
            dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted Exp{cfg.expid} with N_in={N_in}, N_out={N_out}")

# +
# Diagnose trajectories for NaN

_activation_trajs = main.run_experiment()
print(len(_activation_trajs)) # num epochs
print(len(_activation_trajs[0])) # num experiments
print(len(_activation_trajs[0][0])) # (x, y, output)
print(_activation_trajs[0][0]['weights'][0].shape)  # w.shape
print(_activation_trajs[0][0]['weights'][1].shape)  # b.shape
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
# Plot experimental vs modelled neural activity
(_activation_trajs,
 return_exp_activations,
 return_model_activations) = main.run_experiment()
# exp = _experiments[0]
print(return_exp_activations.shape)  # (num_sessions, num_steps, num_recorded_neurons)
print(return_model_activations.shape)  # (num_sessions, num_steps, num_recorded_neurons)
sess=3
vmin = jnp.min(return_exp_activations - return_model_activations)
vmax = jnp.max(return_exp_activations - return_model_activations)
fig, ax = plt.subplots(5, 3, figsize=(8, 8), layout='tight')
im1 = ax[0, 0].imshow(return_exp_activations.mean(2), aspect='auto',
                      cmap='viridis', interpolation='none')
im2 = ax[0, 1].imshow(return_model_activations.mean(2), aspect='auto',
                      cmap='viridis', interpolation='none')
im3 = ax[0, 2].imshow(return_exp_activations.mean(2)
                      - return_model_activations.mean(2),
                      aspect='auto', cmap='bwr', interpolation='none',
                      vmin=vmin,
                      vmax=vmax)
fig.colorbar(im1, ax=ax[0, 0], fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=ax[0, 1], fraction=0.046, pad=0.04)
fig.colorbar(im3, ax=ax[0, 2], fraction=0.046, pad=0.04)
ax[0, 0].set_title("Experimental data")
ax[0, 1].set_title("Modelled data")
ax[0, 0].set_ylabel("Sessions")
ax[0, 0].set_xlabel("Steps")
ax[0, 1].set_xlabel("Steps")
ax[0, 2].set_title("Exp - Model")
ax[0, 2].set_xlabel("Steps")

for sess in range(return_exp_activations.shape[0]):
    ax[1+sess, 0].set_title(f"Exp{sess} data")
    ax[1+sess, 1].set_title(f"Model of Exp{sess} data")
    ax[1+sess, 0].set_ylabel("Steps")
    ax[1+sess, 0].set_xlabel("Neurons")
    ax[1+sess, 1].set_xlabel("Neurons")
    im3 = ax[1+sess, 0].imshow(return_exp_activations[sess], aspect='auto',
                               cmap='viridis', interpolation='none')
    im4 = ax[1+sess, 1].imshow(return_model_activations[sess], aspect='auto',
                               cmap='viridis', interpolation='none')
    fig.colorbar(im3, ax=ax[1+sess, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im4, ax=ax[1+sess, 1], fraction=0.046, pad=0.04)
    im5 = ax[1+sess, 2].imshow(return_exp_activations[sess]
                               - return_model_activations[sess],
                        aspect='auto', cmap='bwr', interpolation='none',
                        vmin=vmin,
                        vmax=vmax)
    fig.colorbar(im5, ax=ax[1+sess, 2], fraction=0.046, pad=0.04)
    ax[1+sess, 2].set_title(f"Exp{sess} - Model of Exp{sess}")
    ax[1+sess, 0].set_ylabel("Steps")
    ax[1+sess, 0].set_xlabel("Neurons")
    ax[1+sess, 2].set_xlabel("Neurons")
plt.show()

# +
# Explore recurrent sparsity parameters
cfg.num_hidden_pre = 50
cfg.num_hidden_post = 50
cfg.recurrent = True
cfg.plasticity_layers = ["rec"]
cfg.feedforward_input_scale = 1
cfg.recurrent_input_scale = 1
last_exp_id = 19
i = 1

for plasticity in [["rec"], ["ff", "rec"]]:
    for input_sparsity in [1, 0.6, 0.3]:
        for ff_sparsity in [1, 0.6, 0.3]:
            for rec_sparsity in [1, 0.6, 0.3]:
        # for ff_scale in [1, 0.5, 0.2]:
        #     for rec_scale in [1, 0.5, 0.2]:
                cfg.plasticity_layers = plasticity
                cfg.postsynaptic_input_sparsity = input_sparsity
                cfg.feedforward_sparsity = ff_sparsity
                cfg.recurrent_sparsity = rec_sparsity
                # cfg.feedforward_input_scale = ff_scale
                # cfg.recurrent_input_scale = rec_scale
                exp_id = last_exp_id + i
                cfg.expid = exp_id
                print(f"\nEXPERIMENT {cfg.expid}:")
                _activation_trajs = main.run_experiment()
                params_dict = {cfg.expid: {"N_in": 50, "N_out": 50,
                                        "plasticity": "+".join(plasticity),
                                           "\ninp_spar": input_sparsity,
                                            "FF_spar": ff_sparsity,
                                            "rec_spar": rec_sparsity,
                                        #    "FF_scale": ff_scale,
                                        #    "rec_scale": rec_scale}}
                                                }}
                recurrent_experiments_config_table.update(params_dict)
                fig = plot_coeff_trajectories(cfg.expid,
                                                recurrent_experiments_config_table,
                                                use_all_81=False)
                fig.savefig(cfg.fig_dir + f"RNN_Exp{cfg.expid} coeff trajectories.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                i += 1
print(recurrent_experiments_config_table)


# +
# Plot xs and ys, optionally evolution of weights

# _activation_trajs, _model_activations, _null_activations = main.run_experiment()

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
epoch = 8
exp = 3
session = 0

# inputs = experiments[exp].data["inputs"]   # (n_sess, n_steps, 1)
# xs = experiments[exp].data["xs"][0]  # (n_sess, n_steps, xdim)
# ys = experiments[exp].data["ys"][0]  # (n_sess, n_steps, ydim)
xs = _activation_trajs[epoch][exp]['xs'][session]
ys = _activation_trajs[epoch][exp]['ys'][session]
w = _activation_trajs[epoch][exp]['weights'][0][session]
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
    # print(f'{jnp.mean(exp.input_weights)=}')
    # print(f'{exp.steps_per_session=}')
    # print(f'{exp.weights[0].shape=}')
    yrange += jnp.max(exp.data["ys"][0, -1]) - jnp.min(exp.data["ys"][0, -1])
print(f'{exp.data["ys"].shape=}')
print(f'{jnp.min(exp.data["xs"])=}, {jnp.max(exp.data["xs"])=}')
print(f'{jnp.min(exp.data["ys"][0, 0])=}, {jnp.max(exp.data["ys"][0, 0])=}')
print(f'{jnp.min(exp.data["ys"][0, -1])=}, {jnp.max(exp.data["ys"][0, -1])=}')
print(f'Average yrange: {yrange / len(experiments)}')

# -

# Plot weights distribution
flat = np.concatenate([exp['weights'][0].ravel() for epoch in _activation_trajs
                       for exp in epoch])
print(flat.shape)
flat = flat[flat < 50]
fig, ax = plt.subplots()
ax.hist(flat, bins=100)
ax.set_xlabel("Synaptic weight")
ax.set_xlim(-5, 5)
plt.show()
