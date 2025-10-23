import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_experiment_results(path, cfg, behavioral_experiments_config_table,
                            show_plots=False):
    """ Plot experiment results:
            - coefficient trajectories,
            - activity and weights trajectories,
            - initial weights heatmaps and traces.

    Args:
        path (str): path to the experiment folder
        cfg (dict): configuration dictionary
        behavioral_experiments_config_table (dict): mapping exp_id -> dict of parameters
        show_plots (bool): whether to show plots interactively
    """
    # Create Plots directory
    save_path = Path(path) / "Plots/"
    save_path.mkdir(parents=True, exist_ok=True)

    fig = plot_coeff_trajectories(cfg.logging.exp_id, path,
                                             behavioral_experiments_config_table,
                                             use_all_81=False)
    fig.savefig(
        save_path / f"Exp_{cfg.logging.exp_id}_coeff_trajectories.png",
        dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    epochs = [0, 50, 100, 150]
    exp = 0
    fig = plot_activity_and_weights_trajectories(path, cfg, epochs, exp, sess=0)
    fig.savefig(
        save_path / f"Exp_{cfg.logging.exp_id}_activity_and_weights (exp {exp}).png",
        dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    mode = "diff"
    fig = plot_init_weights_heatmaps(path, cfg, epochs=epochs, num_exps=5, mode=mode)
    fig.savefig(
        save_path / f"Exp_{cfg.logging.exp_id}_init_weights_heatmaps_{mode}.png")
    if show_plots:
        plt.show()
    plt.close(fig)

    mode = "abs"
    fig = plot_init_weights_traces(path, cfg, mode=mode)
    fig.savefig(
        save_path / f"Exp_{cfg.logging.exp_id}_init_weights_traces_{mode}.png")
    if show_plots:
        plt.show()
    plt.close(fig)

def plot_coeff_trajectories(exp_id, path, params_table, use_all_81=False):
    """
    Plot a single experiment's loss (top), evaluation metrics (middle, if available)
    and coefficient trajectories (bottom).

    This function is 100% ChatGPT 5 generated

    Args:
        exp_id (int): single experiment id
        path (str): path to the experiment folder
        params_table (dict): mapping exp_id -> dict of parameters for subplot title
        use_all_81 (bool): whether to plot 81 coefficients (with rewards) or just 27
    """

    # single file path for the single experiment
    fpath = os.path.join(path, f"Exp_{exp_id}_results.csv")
    # if not os.path.exists(fpath):
    #     new_dir = os.path.join(os.path.dirname(path.rstrip('/\\')), f"Exp_{exp_id}")
    #     fpath = os.path.join(new_dir, f"Exp_{exp_id}_results.csv")

    # highlight a few columns (if they exist)
    highlight = {"F_1100", "F_0210"}  # TODO use generation_plasticity config

    # --- read data ---
    df = pd.read_csv(fpath)

    # top = Loss, bottom = coeff trajectories[, middle = evaluation metrics]
    has_eval = 'train_loss_median' in df.columns

    # For 27-mode we keep existing layout;
    # for 81-mode we will recreate below with 3 coeff subplots
    if not use_all_81:
        if has_eval:
            fig, axs = plt.subplots(3, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [1, 2, 2]},
                                    sharex=True,
                                    layout='tight')
            top_ax, eval_ax, coeff_ax = axs
        else:
            fig, axs = plt.subplots(2, 1, figsize=(12, 7))
            top_ax, coeff_ax = axs
            eval_ax = None
    else:
        # placeholder; we'll create the desired layout
        # (loss + optional eval + 3 coeff subplots)
        top_ax = eval_ax = coeff_ax = None

    # --- Top: loss subplot (backwards-compatible) ---
    x_epochs = df['epoch'] if 'epoch' in df.columns else np.arange(len(df))

    # If we didn't yet create axes (use_all_81 True), create them now properly
    if use_all_81:
        if has_eval:
            fig, axs = plt.subplots(5, 1, figsize=(12, 13),
                                    gridspec_kw={'height_ratios': [1, 1, 2, 2, 2]},
                                    sharex=True)
            top_ax, eval_ax, coeff0_ax, coeff1_ax, coeff2_ax = axs
            coeff_axes = [coeff0_ax, coeff1_ax, coeff2_ax]
        else:
            fig, axs = plt.subplots(4, 1, figsize=(12, 13),
                                    gridspec_kw={'height_ratios': [1, 2, 2, 2]},
                                    sharex=True)
            top_ax, coeff0_ax, coeff1_ax, coeff2_ax = axs
            eval_ax = None
            coeff_axes = [coeff0_ax, coeff1_ax, coeff2_ax]
    else:
        # we already have coeff_ax for 27-mode
        coeff_axes = [coeff_ax]

    # plot loss on top_ax
    if 'train_loss_median' in df.columns and 'test_loss_median' in df.columns:
        top_ax.plot(x_epochs, df['train_loss_median'],
                    color='blue', label='train_loss_median')
        top_ax.plot(x_epochs, df['test_loss_median'],
                    color='red', label='test_loss_median')
    elif all(col in df.columns for col in ['train_loss_mean','train_loss_std',
                                           'test_loss_mean','test_loss_std']):
        top_ax.plot(x_epochs, df['train_loss_mean'],
                    color='blue', label='train_loss')
        top_ax.fill_between(x_epochs,
                            df['train_loss_mean'] - df['train_loss_std'],
                            df['train_loss_mean'] + df['train_loss_std'],
                            color='blue', alpha=0.2)
        top_ax.plot(x_epochs, df['test_loss_mean'],
                    color='red', label='test_loss')
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
        colors = [cmap(0.5)] if n == 1 else [cmap(t) for t in np.linspace(0, 1, n)]
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

    # --- Special handling for use_all_81:
    # three subplots, same y-limits, no markers, no jitter ---
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
                ax.plot(x_vals, df[c].values, label=label_map.get(c, c),
                        linewidth=lw, linestyle=ls, color=color)
            ax.set_ylim(y_lim)
            ax.grid(True)
            ax.set_ylabel('')

            # build legend for this axis with 27 base entries (no markers)
            legend_handles = []
            legend_labels = []
            for key3 in base_keys:
                # find column with r==rexp that matches key3;
                # if not found, pick representative with r==0 for label/color
                target = next((cc for cc in data_cols
                               if parsed[cc][:3] == key3 and parsed[cc][3] == rexp),
                               None)
                if target is None:
                    # fall back to any (a,b,w) representative
                    # (prefer r=0 to keep color consistent)
                    target = next((cc for cc in data_cols
                                   if parsed[cc][:3] == key3 and parsed[cc][3] == 0),
                                   None)
                if target is None:
                    # ultimate fallback, pick any matching (a,b,w)
                    target = next((cc for cc in data_cols
                                   if parsed[cc][:3] == key3),
                                  None)
                if target is None:
                    legend_handles.append(Line2D([0],[0], color='none'))
                    legend_labels.append('')
                else:
                    # line-only proxy (no marker)
                    color = color_map.get(target, 'k')
                    ls = linestyle_map.get(parsed[target][2], '-')
                    lw = 3 if target in highlight else 2
                    legend_handles.append(Line2D([0],[0], color=color,
                                                 lw=lw, linestyle=ls))
                    legend_labels.append(pretty_label(target))

            # place legend beneath this coefficient subplot
            ax.legend(legend_handles, legend_labels, loc='lower center',
                      bbox_to_anchor=(0.5, -0.65),
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
            ax.set_title(f"Exp_{exp_num}: {param_str}", fontsize=12)
        else:
            ax.set_title(f"Exp_{exp_num}", fontsize=12)

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

    return fig

def plot_init_weights_traces(path, cfg, sample_size=10, num_exps=5, mode="abs"):
    """ Plot learning trajectories of initial weights
    across epochs for multiple experiments.

    Args:
        path: path to the experiment folder
        cfg: configuration object
        sample_size: number of weights to sample for visualization
        num_exps: number of experiments to plot
        mode:
            'abs' to plot absolute weights and ground-truth,
            'diff' to plot difference from ground-truth
    Returns:
        fig: matplotlib figure object
    """

    def plot_layer_weight_traces(x_ax, y_ax, exp, layer):
        ws_gen = exp.w_init_gen[layer]['w'].flatten()
        ws_train = [trajectories[epoch][i]['init_weights'][layer]['w']
                   for epoch in epochs]
        ws_train = np.array(ws_train).reshape(num_epochs, -1)

        # Randomly sample a subset of weights for visualization
        if ws_train.shape[1] > sample_size:
            indices = np.random.choice(ws_train.shape[1], size=sample_size,
                                       replace=False)
            ws_train = ws_train[:, indices]
            ws_gen = ws_gen[indices]

        n_weights = ws_gen.shape[0]
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, n_weights))

        if mode == 'abs':
            # plot ground-truth generation weights
            for w_gen, color in zip(ws_gen, colors, strict=False):
                ax[x_ax, y_ax].hlines(w_gen, epochs[0], epochs[-1],
                                      colors=color, alpha=0.5, linestyles='dashed')
        elif mode == 'diff':
            ws_train = ws_train - ws_gen[np.newaxis, :]

        # plot learning trajectories for each weight
        for w_trace, color in zip(ws_train.T, colors, strict=False):
            ax[x_ax, y_ax].plot(epochs, w_trace, color=color, linewidth=1.0)

        ax[x_ax, y_ax].set_xticks(epochs)
        if x_ax == 0:
            ax[x_ax, y_ax].set_title(f"{layer} init weights")
        ax[x_ax, y_ax].set_xlabel("Epoch")
        if y_ax == 0:
            ax[x_ax, y_ax].set_ylabel(f"Experiment {i}")

    experiments = utils.load_generated_experiments(path, cfg, mode="train")
    num_exps = min(num_exps, len(experiments))
    trajectories = utils.load_hdf5(path + f"Exp_{cfg.logging.exp_id}_trajectories.h5")
    epochs = list(trajectories.keys())

    num_epochs = len(epochs)
    epochs = np.array(epochs)
    layers = list(trajectories[epochs[0]][0]['init_weights'].keys())
    num_layers = len(layers)
    fig, ax = plt.subplots(num_exps, num_layers,
                           figsize=(10, 3 * num_exps), layout='tight')
    ax = ax.reshape(num_exps, num_layers)  # ensure 2D array
    for i in range(num_exps):
        exp = experiments[i]
        for j, layer in enumerate(layers):
            plot_layer_weight_traces(i, j, exp, layer)

    return fig

def plot_init_weights_heatmaps(path, cfg, epochs, num_exps=5, mode="diff"):
    """ Plot heatmaps of learned initial weights at selected epochs.

    Args:
        path: path to the experiment folder
        cfg: configuration object
        epochs: list of epochs to plot
        num_exps: number of experiments to plot
        mode:
            'abs' to plot absolute weights and ground-truth,
            'diff' to plot difference from ground-truth
    Returns:
        fig: matplotlib figure object
    """

    def plot_layer_weight_traces(row, col, layer, epoch, w_gen):
        w_train = trajectories[epoch][i]['init_weights'][layer]['w']
        if mode == 'diff':
            w_train = w_train - w_gen

        im = ax[row, col].imshow(w_train, vmin=v_min, vmax=v_max, aspect='equal',
                                 cmap='viridis', interpolation='none', origin='lower')
        title = ""
        if row == 0:
            title += f"Epoch {epoch}\nInit weights"
            if mode == 'diff':
                title += " - diff from GT"
        if row % num_layers == 0:
            title += f"\nExperiment {i}"
        ax[row, col].set_title(title)
        if col == 0:
            ax[row, col].set_ylabel(f"{layer} layer")

        return im

    experiments = utils.load_generated_experiments(path, cfg, mode="train")
    num_exps = min(num_exps, len(experiments))
    trajectories = utils.load_hdf5(path + f"Exp_{cfg.logging.exp_id}_trajectories.h5")
    recorded_epochs = list(trajectories.keys())

    epochs = [epoch for epoch in epochs if epoch in recorded_epochs]
    num_epochs = len(epochs)
    if mode == 'abs':
        num_epochs += 1  # for ground-truth column
    epochs = np.array(epochs)

    layers = list(trajectories[epochs[0]][0]['init_weights'].keys())
    num_layers = len(layers)

    n_rows = num_exps * num_layers
    n_cols = num_epochs + 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows),
                           gridspec_kw={'width_ratios': [1.0]*num_epochs + [0.03]},
                           layout='constrained')
    ax = ax.reshape(n_rows, n_cols)  # ensure 2D array even if n_cols=1
    for i in range(num_exps):
        exp = experiments[i]
        for j, layer in enumerate(layers):
            w_gen = exp.w_init_gen[layer]['w']
            # Set color scale limits based on ground-truth weights.
            # Allow training weights twice the range of gen weights
            v_min, v_max = np.min(w_gen)*2, np.max(w_gen)*2

            for k, epoch in enumerate(epochs):
                im = plot_layer_weight_traces(num_layers*i+j, k, layer, epoch, w_gen)
                # ax[3*i+j, k].set_clim(v_min, v_max)
            if mode == 'abs':
                # Last column is colorbar, plot ground-truth weights to second last
                ax[num_layers*i+j, -2].imshow(w_gen, vmin=v_min, vmax=v_max,
                                                   aspect='equal', cmap='viridis',
                                                   interpolation='none', origin='lower')
                if i == 0 and j == 0:
                    ax[num_layers*i+j, -2].set_title("Ground-truth init weights")

            mappable = plt.cm.ScalarMappable(cmap=im.get_cmap(), norm=im.norm)
            mappable.set_array([])
            plt.colorbar(mappable, cax=ax[num_layers*i+j, -1], fraction=0.02, pad=0.001)

    return fig

def plot_activity_and_weights_trajectories(path, cfg, epochs, exp=0, sess=0):
    """ Plot trajectories of neural activity and weight
    of selected experiment at different epochs.

    Args:
        path: Path to the experiment data.
        cfg: Configuration object.
        epochs: List of epochs to plot.
        exp: Experiment index.
        sess: Session index.

    Returns:
        fig: Matplotlib figure object.
    """
    def plot_epoch_trajectories(epoch, col_i):
        xs = trajectories[epoch][exp]['xs'][sess].T
        ys = trajectories[epoch][exp]['ys'][sess].T
        n_steps = xs.shape[1]

        xs_ax = ax[0, col_i].imshow(xs, aspect='auto',
                            cmap='viridis', interpolation='none', origin='lower')
        ax[0, col_i].set_title(f'Epoch {epoch}\nX activity')
        divider = make_axes_locatable(ax[0, col_i])
        cax = divider.append_axes("right", size="1.5%", pad=0.03)
        fig.colorbar(xs_ax, cax=cax)
        ax[0, col_i].set_ylabel('Neurons')
        ax[0, col_i].set_xlabel('Steps')

        ys_ax = ax[1, col_i].imshow(ys, aspect='auto',
                            cmap='viridis', interpolation='none', origin='lower')
        ax[1, col_i].set_title('Y activity')
        # fig.colorbar(ys_ax, ax=ax[1, col_i], fraction=0.04, pad=0.01, location=)
        divider = make_axes_locatable(ax[1, col_i])
        cax = divider.append_axes("right", size="1.5%", pad=0.03)
        fig.colorbar(ys_ax, cax=cax)
        ax[1, col_i].set_ylabel('Neurons')
        ax[1, col_i].set_xlabel('Steps')

        output = trajectories[epoch][exp]['outputs'][sess]
        d = trajectories[epoch][exp]['decisions'][sess]
        r = trajectories[epoch][exp]['rewards'][sess]
        ax[2, col_i].plot(output, color='gray', linewidth=2, label='output', alpha=0.5)
        # Plots decisions as dots. y=0 for no reward, y=1 for reward
        idx = np.flatnonzero(np.asarray(d) == 1)
        ax[2, col_i].scatter(idx, np.where(np.asarray(r)[idx] == 1, 1, 0),
                             s=10, c='k', zorder=10)

        row_2_title = 'Output, decisions and rewards'

        if 'w_ff' in trajectories[epoch][exp]['weights']:
            wff = trajectories[epoch][exp]['weights']['w_ff'][sess]
            traces_ff = wff.reshape(wff.shape[0], -1).T

            # Construct colormap based on input type
            if cfg.experiment.input_type == 'random':
                colors = plt.get_cmap('viridis')(np.linspace(0, 1, traces_ff.shape[0]))
            elif cfg.experiment.input_type == 'task':
                colors_pl = plt.get_cmap('cool')(np.linspace(
                    0, 1, cfg.experiment.num_place_neurons * wff.shape[-1]))
                colors_vis = plt.get_cmap('autumn')(np.linspace(
                    0, 1, (wff.shape[-2]-cfg.experiment.num_place_neurons-1)
                    * wff.shape[-1]))
                colors_v = plt.get_cmap("Greens")(np.linspace(
                    0, 1, cfg.experiment.num_velocity_neurons * wff.shape[-1]))
                colors = np.concatenate([colors_pl, colors_vis, colors_v])

            # repeat each presyn color for all posts so it matches traces_ff rows
            colors_traces = np.repeat(colors, wff.shape[-1], axis=0)
            for trace, color in zip(traces_ff, colors_traces, strict=False):
                ax[2, col_i].plot(trace, color=color, alpha=0.1)

            row_2_title += '.\nFeedforward weights traces'

        ax[2, col_i].set_title(row_2_title)
        ax[2, col_i].set_xlabel('Steps')

        if 'w_rec' in trajectories[epoch][exp]['weights']:
            wrec = trajectories[epoch][exp]['weights']['w_rec'][sess]
            traces_rec = wrec.reshape(wrec.shape[0], -1).T

            colors = plt.get_cmap('viridis')(np.linspace(0, 1, traces_rec.shape[0]))
            for trace, color in zip(traces_rec, colors, strict=False):
                ax[3, col_i].plot(trace, color=color, alpha=0.1)

            ax[3, col_i].set_title('Recurrent weights traces')
            ax[3, col_i].set_xlabel('Steps')

        for irow in range(ax.shape[0]):
            ax[irow, col_i].set_xlim(-0.5, n_steps - 0.5)

    trajectories = utils.load_hdf5(path + f"Exp_{cfg.logging.exp_id}_trajectories.h5")
    recorded_epochs = list(trajectories.keys())
    epochs = [epoch for epoch in epochs if epoch in recorded_epochs]
    n_cols = len(epochs)
    n_rows = 3
    if 'w_rec' in trajectories[epochs[0]][exp]['weights']:
        n_rows += 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows),
                           layout='tight')
    ax = ax.reshape(n_rows, n_cols)  # ensure 2D array even if n_cols=1

    for col_i, epoch in enumerate(epochs):
        plot_epoch_trajectories(epoch, col_i)

    return fig
