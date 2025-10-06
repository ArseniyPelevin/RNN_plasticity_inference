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

# jax.config.update('jax_log_compiles', True)
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

    # --- read data ---
    df = pd.read_csv(fpath)

    # top = Loss, bottom = coeff trajectories[, middle = evaluation metrics]
    has_eval = 'train_loss_median' in df.columns

    # For 27-mode we keep existing layout; for 81-mode we will recreate below with 3 coeff subplots
    if not use_all_81:
        if has_eval:
            fig, axs = plt.subplots(3, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [1, 2, 2]}, sharex=True,
                                    layout='tight')
            top_ax, eval_ax, coeff_ax = axs
        else:
            fig, axs = plt.subplots(2, 1, figsize=(12, 7))
            top_ax, coeff_ax = axs
            eval_ax = None
    else:
        # placeholder; we'll create the desired layout (loss + optional eval + 3 coeff subplots)
        top_ax = eval_ax = coeff_ax = None

    # --- Top: loss subplot (backwards-compatible) ---
    x_epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))

    # If we didn't yet create axes (use_all_81 True), create them now properly
    if use_all_81:
        if has_eval:
            fig, axs = plt.subplots(5, 1, figsize=(12, 13),
                                    gridspec_kw={'height_ratios': [1, 1, 2, 2, 2]}, sharex=True)
            top_ax, eval_ax, coeff0_ax, coeff1_ax, coeff2_ax = axs
            coeff_axes = [coeff0_ax, coeff1_ax, coeff2_ax]
        else:
            fig, axs = plt.subplots(4, 1, figsize=(12, 13),
                                    gridspec_kw={'height_ratios': [1, 2, 2, 2]}, sharex=True)
            top_ax, coeff0_ax, coeff1_ax, coeff2_ax = axs
            eval_ax = None
            coeff_axes = [coeff0_ax, coeff1_ax, coeff2_ax]
    else:
        # we already have coeff_ax for 27-mode
        coeff_axes = [coeff_ax]

    # plot loss on top_ax
    if 'train_loss_median' in df.columns and 'test_loss_median' in df.columns:
        top_ax.plot(x_epochs, df['train_loss_median'], color='blue', label='train_loss_median')
        top_ax.plot(x_epochs, df['test_loss_median'], color='red', label='test_loss_median')
    elif all(col in df.columns for col in ['train_loss_mean','train_loss_std',
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

    top_ax.set_title("Train and Test Loss", fontsize=12)
    top_ax.legend(loc='upper right')
    top_ax.grid(True)
    top_ax.set_ylabel('Loss')

    # --- Middle: evaluation metrics subplot (if available) ---
    if eval_ax is not None:
        metrics = ['PDE_F_neural', 'PDE_T_neural', 'PDE_W_neural',
                   'R2_F_y', 'R2_F_w', 'R2_T_y', 'R2_T_w', 'R2_W_y', 'R2_W_w']

        for metric in metrics:
            if 'PDE' in metric:
                line_style = '-'
                k = 1
            elif 'R2' in metric:
                continue
                k = 100
                if '_y' in metric:
                    line_style = '--'
                elif '_w' in metric:
                    line_style = ':'

            if '_F_' in metric:
                color = 'blue'
            elif '_T_' in metric:
                color = 'purple'
            elif '_W_' in metric:
                color = 'red'
            if metric in df.columns:
                eval_ax.plot(x_epochs, df[metric]*k, label=metric,
                             linestyle=line_style, color=color)
        eval_ax.legend(loc='center right', fontsize=8)
        eval_ax.set_ylabel('Percent deviance / R2 * 100')
        eval_ax.grid(True)
        eval_ax.set_title("Evaluation Metrics", fontsize=12)

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

    # Fallback: if nothing found
    if not data_cols:
        coeff_axes[0].text(0.5, 0.5, "No coefficient columns found in CSV",
                            ha='center', va='center')
        plt.show()
        return

    # parse suffixes
    parsed = {}
    for c in data_cols:
        suffix = str(c).split('_')[-1]
        a, b, w, r = map(int, list(suffix))
        parsed[c] = (a, b, w, r)

    # deterministic ordering key
    def col_key(col):
        s = str(col).split('_')[-1]
        return (tuple(map(int, list(s))) if len(s) == 4 and s.isdigit()
                else (0, 0, 0, 0))

    # assign colors grouped like the 27-mode so r=0 colors match the old plot
    # build ordered list of base (a,b,w) keys in deterministic order
    base_keys = []
    seen = set()
    for c in sorted(data_cols, key=col_key):
        a,b,w,r = parsed[c]
        key3 = (a,b,w)
        if key3 not in seen:
            seen.add(key3)
            base_keys.append(key3)

    # assign colors per base_key but grouped by w (to mimic 27-mode coloring)
    color_map = {}
    for w_val in sorted({k[2] for k in base_keys}):
        group_keys = [k for k in base_keys if k[2] == w_val]
        n = len(group_keys)
        cmap = plt.get_cmap('Set1')
        if n <= 1:
            colors = [cmap(0.5)]
        else:
            colors = [cmap(t) for t in np.linspace(0, 1, n)]
        for key3, colcolor in zip(group_keys, colors, strict=False):
            # assign this color to every column that matches the (a,b,w) base key
            for cc in data_cols:
                if parsed[cc][:3] == key3:
                    color_map[cc] = colcolor

    # fallback: ensure every column has a color
    for c in data_cols:
        if c not in color_map:
            color_map[c] = plt.get_cmap('tab10')(0)

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
    x_vals = np.asarray(x)

    # --- Special handling for use_all_81: three subplots, same y-limits, no markers, no jitter ---
    if use_all_81:
        # compute global y limits across all data columns
        all_vals = np.hstack([df[c].values for c in data_cols])
        ymin, ymax = np.min(all_vals), np.max(all_vals)
        pad = max(1e-8, 0.05 * (ymax - ymin)) if ymax > ymin else 0.1
        y_lim = (ymin - pad, ymax + pad)

        # build ordered 27 base keys (a,b,w) preserving original order used in 27-mode
        base_keys = []
        seen = set()
        for c in sorted(data_cols, key=col_key):
            a,b,w,r = parsed[c]
            key3 = (a,b,w)
            if key3 not in seen:
                seen.add(key3)
                base_keys.append(key3)

        # plot each r in its own axis
        # allocate extra vertical space so per-axis legends can sit between subplots
        fig.subplots_adjust(hspace=0.7)
        for rexp, ax in enumerate(coeff_axes):
            for c in sorted(data_cols, key=col_key):
                a,b,w,r = parsed[c]
                if r != rexp:
                    continue
                ls = linestyle_map.get(w, '-')
                lw = 3 if c in highlight else 2
                color = color_map.get(c, 'k')
                # no jitter, no markers
                ax.plot(x_vals, df[c].values, label=label_map.get(c, c), linewidth=lw, linestyle=ls, color=color)
            ax.set_ylim(y_lim)
            ax.grid(True)
            ax.set_ylabel('')

            # build legend for this axis with 27 base entries (no markers)
            legend_handles = []
            legend_labels = []
            for key3 in base_keys:
                # find column with r==rexp that matches key3; if not found, pick representative with r==0 for label/color
                target = next((cc for cc in data_cols if parsed[cc][:3] == key3 and parsed[cc][3] == rexp), None)
                if target is None:
                    # fall back to any (a,b,w) representative (prefer r=0 to keep color consistent)
                    target = next((cc for cc in data_cols if parsed[cc][:3] == key3 and parsed[cc][3] == 0), None)
                if target is None:
                    # ultimate fallback, pick any matching (a,b,w)
                    target = next((cc for cc in data_cols if parsed[cc][:3] == key3), None)
                if target is None:
                    legend_handles.append(Line2D([0],[0], color='none'))
                    legend_labels.append('')
                else:
                    # line-only proxy (no marker)
                    color = color_map.get(target, 'k')
                    ls = linestyle_map.get(parsed[target][2], '-')
                    lw = 3 if target in highlight else 2
                    legend_handles.append(Line2D([0],[0], color=color, lw=lw, linestyle=ls))
                    legend_labels.append(pretty_label(target))

            # place legend beneath this coefficient subplot
            ax.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.65),
                      ncol=9, fontsize=9, frameon=False)

        # set title on the middle coeff axis (or topmost coeff axis)
        basename = os.path.basename(fpath)
        exp_num = exp_id
        if exp_num is not None and exp_num in params_table:
            p = params_table[exp_num]
            param_str = ', '.join(f'{key}={value}' for key, value in p.items())
            coeff_axes[0].set_title(f"{basename[:-4]}: {param_str}", fontsize=12)
        else:
            coeff_axes[0].set_title(basename, fontsize=12)

    else:
        # --- existing 27-parameter plotting logic (unchanged) ---
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
        ax = coeff_axes[0]

        # jitter in axis units: small horizontal offset for R groups
        jitter = 1.0
        x_vals = np.asarray(x)

        # plot lines + markers according to R (original behavior)
        for c in sorted(data_cols, key=col_key):
            a, b, wexp, rexp = parsed[c]
            lw = 3 if c in highlight else 2
            ls = linestyle_map.get(wexp, '-')
            color = color_map.get(c, 'k')
            # shift both line and markers by rexp * jitter (in axis units)
            x_plot = x_vals + (rexp * jitter)
            ax.plot(x_plot, df[c], label=label_map.get(c, c), linewidth=lw, linestyle=ls, color=color)
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

        # older 27-parameter legend (unchanged)
        legend_handles = []
        legend_labels = []
        for key3 in base_keys:
            target = next((cc for cc in data_cols
                           if parsed[cc][:3] == key3
                           and parsed[cc][3] == 0), None)
            if target is None:
                target = next((cc for cc in data_cols
                               if parsed[cc][:3] == key3), None)
            if target is None:
                legend_handles.append(Line2D([0], [0], color='none'))
                legend_labels.append('')
            else:
                legend_handles.append(proxy_handle(target, 0))
                legend_labels.append(label_map.get(target, target))
        fig.subplots_adjust(bottom=0.30, hspace=0.50, top=0.92)

        ncol = 9
        fig.legend(legend_handles, legend_labels,
                   loc='lower center', bbox_to_anchor=(0.5, -0.10),
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

    # set xticks on the bottom-most axis
    coeff_axes[-1].set_xticks(tick_positions)
    coeff_axes[-1].set_xticklabels([str(int(v)) if float(v).is_integer() else str(v)
                                     for v in tick_positions])

    plt.show()

    return fig



# +
# Set parameters and run experiment
# training = reload(training)
# experiment = reload(experiment)
# model = reload(model)

# parameters table to include in subplot titles
# import json
# with open("feedforward_experiments_config_table.json", "r") as f:
#     feedforward_experiments_config_table = json.load(f)

behavioral_experiments_config_table = {
     # 1: {'recurrent': True, 'plasticity': "ff, rec", 'train_w': "w_rec, w_ff",
     #       "\nN_in": 50, "N_out": 50, 'init_spars': 'ff: 0.5, rec: 0.5',
     #       '\ninit_w_mean': 'ff=2, rec=-1', 'init_w_std': 'ff=1, rec=1',
     #       '\nx': '{0,1}', 'n_epochs': 250, 'seed': 189, "null_model": "zeros"},
     3: {"N_in": 50, "N_out": 50, 'scale_by_N_inputs': False,},
     4: {"N_in": 50, "N_out": 50, 'scale_by_N_inputs': True,},
     7: {"N_in": 50, "N_out": 50, 'plasticity': "ff, rec",
         '\nBad seed': "don't use"},
     8: {"N_in": 50, "N_out": 50, 'plasticity': "ff, rec",
         '\nscale_by_N_inputs': True, 'init_weights_std': 'ff=0.1, rec=0.1'},
     9: {"N_in": 50, "N_out": 50, 'plasticity': "ff, rec",
         '\nscale_by_N_inputs': False, 'init_weights_std': 'ff=0.1, rec=0.1'},
    10: {"N_in": 50, "N_out": 50, 'plasticity': "ff, rec",
         '\nscale_by_N_inputs': False, 'init_weights_std': 'ff=Kaiming, rec=Kaiming'},
    11: {"N_in": 50, "N_out": 50, 'plasticity': "ff, rec",
         '\nscale_by_N_inputs': True, 'rule': '$x^{2}y-yw$',},
    12: {"input_type": 'task', 'plasticity': "ff, rec", "sparsity": 1,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid"},
    13: {"input_type": 'task', 'plasticity': "ff, rec", "sparsity": 1,
         '\nfeedforward_input_scale': 0.5, 'rule': '$xy-y^{2}w$', "activation": "sigmoid"},
    14: {"input_type": 'task', 'plasticity': "ff, rec", "sparsity": 1,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid(y-1)"},
    15: {"input_type": 'task', 'plasticity': "ff, rec", "input_sparsity": 0.5, 'velocity': True,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid(y-1)"},
    16: {"input_type": 'task', 'plasticity': "ff, rec", "input_sparsity": 0.5, 'velocity': False,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid(y-1)",
         "x_noise": 0.05,},
    17: {"input_type": 'task', 'plasticity': "ff, rec", "input_sparsity": 0.5, 'velocity': False,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid(y-1)",
         "x_noise": 0.1,},
    18: {"input_type": 'task', 'plasticity': "ff, rec", "input_sparsity": 0.3, 'velocity': False,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid(y-1)",
         "x_noise": 0.1,},
    19: {"input_type": 'task', 'plasticity': "ff, rec", "input_sparsity": 0.3, 'velocity': False,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid(y)",
         "x_noise": 0.1,},
    20: {"input_type": 'task', 'plasticity': "ff, rec", "inp_spars_gen": 0.3, "inp_spars_train": 1, 'velocity': False,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid(y-1)",
         "x_noise": 0.1,},
    21: {"input_type": 'task', 'plasticity': "ff, rec", "input_sparsity": 0.3, 'velocity': True,
         '\nfeedforward_input_scale': 1, 'rule': '$xy-y^{2}w$', "activation": "sigmoid(y-1)",
         "x_noise": 0.1,},
    22: {"input_type": 'task', 'plasticity': "ff, rec", "input_sparsity": 0.3, 'velocity': True,
         '\nfeedforward_input_scale': 1, 'rule': '$xr-w$', "activation": "sigmoid(y-1)",
         "x_noise": 0.1,},
    24: {'rule': "$xyr+0.3xy-0.3y^{2}w$", "trainable_w": "w_rec, w_ff, w_out", "min_lick_pr": 0.01},
    25: {'rule': "$xyr+0.3xy-0.3y^{2}w$", "trainable_w": "w_rec, w_ff, w_out", "min_lick_pr": 0.05,
         '\ntrials_per_sess': 20,},
}

cfg = main.create_config()

cfg.num_exp_train = 25
cfg.num_exp_test = 5
cfg.num_test_restarts = 5

cfg.fit_data = 'reinforcement'

cfg.recurrent = True
cfg.trainable_init_weights = ["w_rec", "w_ff", "w_out"]
cfg.plasticity_layers = ["ff", "rec"]
cfg.postsynaptic_input_sparsity_generation = 0.3
cfg.postsynaptic_input_sparsity_training = 0.3
cfg.feedforward_sparsity_generation = 1
cfg.feedforward_sparsity_training = 1
cfg.recurrent_sparsity_generation = 1
cfg.recurrent_sparsity_training = 1

cfg.presynaptic_firing_mean = 0
cfg.presynaptic_noise_std = 0.1

# cfg.init_weights_sparsity_generation = {'ff': 0.5, 'rec': 0.5}
# cfg.init_weights_mean_generation = {'ff': 2, 'rec': -1, 'out': 0}
# cfg.init_weights_std_generation = {'ff': 1, 'rec': 1, 'out': 0}
# cfg.init_weights_std_training = {'ff': 1, 'rec': 1, 'out': 0}
cfg.init_weights_sparsity_generation = {'ff': 1, 'rec': 1}
cfg.init_weights_mean_generation = {'ff': 0, 'rec': 0, 'out': 0}
cfg.init_weights_std_generation = {'ff': 0.1, 'rec': 0.1, 'out': 1}
cfg.init_weights_std_training = {'ff': 0.1, 'rec': 0.1, 'out': 1}

# Exp57 = scaling by number of inputs, ff sparsity = 1
cfg.expid = 26
cfg.num_hidden_pre = 50
cfg.num_hidden_post = 30

cfg.mean_num_sessions = 1
cfg.std_num_sessions = 0
cfg.mean_trials_per_session = 20
cfg.std_trials_per_session = 0
# cfg.mean_steps_per_trial = 50
# cfg.std_steps_per_trial = 0
cfg.mean_trial_time = 29
cfg.std_trial_time = 0
cfg.dt = 1
cfg.feedforward_input_scale = 1

cfg.num_epochs = 250
cfg.learning_rate = 3e-2
cfg.lick_cost = 1/13
cfg.input_type = 'task'
cfg.do_evaluation = True
cfg.log_expdata = True
cfg.log_trajectories = True

cfg.generation_plasticity = "1X1Y1W0R1+0.31X1Y1W0R0-0.3X0Y2W1R0"  # Oja's

params, expdata, _activation_trajs, _losses_and_r2s = main.run_experiment(cfg)

fig = plot_coeff_trajectories(cfg.expid, behavioral_experiments_config_table,
                              use_all_81=True)
fig.savefig(cfg.fig_dir + f"Beh_Exp{cfg.expid} coeff trajectories.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
# +
# Plot example input
import experiment

cfg.mean_num_sessions = 3  # Number of sessions/days per experiment/trajectory/animal
cfg.std_num_sessions = 0  # Standard deviation of sessions/days per animal
cfg.mean_trials_per_session = 3  # Number of trials/runs in each session/day
cfg.std_trials_per_session = 0
cfg.mean_steps_per_trial = 50  # Mean number of time steps in each trial/run
cfg.std_steps_per_trial = 0  # Standard deviation of time steps in each trial/run

shapes, step_mask = experiment.define_experiments_shapes(jax.random.PRNGKey(0), 1, cfg)
inputs = experiment.generate_inputs(jax.random.PRNGKey(0),
                                 shapes, step_mask[0],
                                 cfg, exp_i=0)

t, v, pos, cue_at_time, rewarded_pos = [v[0] for v in inputs.values()]
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(cue_at_time*50, label='Visual cue', linewidth=2)
ax.plot(pos, label='Position (cm)', linewidth=2)
# ax.plot(t, segment_at_time*10, label='10 cm segment', linewidth=2)
ax.plot(v*10+3, label='Velocity x 10 (cm/s)',
        linewidth=2, linestyle='--')
ax.plot(rewarded_pos*10, label='Rewarded position',
        linewidth=2, linestyle=':', color='green')
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (cm)')
ax.set_title('2AFC Task Structure with Visual Cues and Rewards\nNear trial')
plt.show()

# +
# Explore different n_steps
cfg.init_weights_sparsity_generation = {'ff': 0.5, 'rec': 0.5}
cfg.init_weights_mean_generation = {'ff': -0.2, 'rec': 0.3, 'out': 0}
cfg.init_weights_std_generation = {'ff': 1, 'rec': 1, 'out': 0}
cfg.init_weights_std_training = {'ff': 0.1, 'rec': 0.1, 'out': 0.1}

# Exp57 = scaling by number of inputs, ff sparsity = 1
cfg.expid = 180
cfg.num_hidden_pre = 1000
cfg.num_hidden_post = 1000

cfg.num_epochs = 300

for steps in [5, 10, 15, 20, 50, 100, 150]:
    cfg.mean_steps_per_trial = steps
    cfg.expid += 1
    _activation_trajs, _losses_and_r2s = main.run_experiment(cfg)

    param_table = {
        cfg.expid: {
            'recurrent': True, 'plasticity': "ff, rec", 'train_w': "w_rec, w_ff",
           "\nN_in": 1000, "N_out": 1000, 'init_spars': 'ff: 0.5, rec: 0.5',
           '\ninit_w_mean': 'ff=-0.2, rec=0.3', 'init_w_std': 'ff=1, rec=1',
           '\nn_steps': steps}}

    fig = plot_coeff_trajectories(cfg.expid, param_table,
                                use_all_81=False)
    fig.savefig(cfg.fig_dir + f"RNN_Exp{cfg.expid} coeff trajectories.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

# +
# Plot histograms of losses and R2 values for all experiments

fig, ax = plt.subplots(2, 2, figsize=(10, 10), layout='tight')
ax = ax.flatten()
for i, metric in enumerate(['loss', 'MSE', 'r2_y', 'r2_w']):
    dF = _losses_and_r2s['F'][metric].ravel()
    dN = _losses_and_r2s['N'][metric].ravel()

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
