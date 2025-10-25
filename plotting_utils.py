from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_experiment_results(path, params_table=None,
                            epochs=None, num_epochs=4,
                            show_plots=False, save_plots=False):
    """ Plot experiment results:
            - coefficient trajectories,
            - activity and weights trajectories,
            - initial weights heatmaps and traces.

    Args:
        path (str): path to the experiment folder
        cfg (dict): configuration dictionary
        behavioral_experiments_config_table (dict): mapping exp_id -> dict of parameters
        epochs (list): list of epochs to plot
        num_epochs (int): number of epochs to plot if epochs is None
        show_plots (bool): whether to show plots interactively
    """
    # Create Plots directory
    save_path = Path(path) / "Plots/"
    save_path.mkdir(parents=True, exist_ok=True)

    cfg = utils.load_config(path)
    expdata = utils.load_expdata(path)
    experiments = utils.load_generated_experiments(path, cfg, mode="train")
    trajectories = utils.load_hdf5(path + f"Exp_{cfg.logging.exp_id}_trajectories.h5")

    fig = plot_coeff_trajectories(data=(cfg, expdata), params_table=params_table)
    if save_plots:
        fig.savefig(
            save_path / f"Exp_{cfg.logging.exp_id}_coeff_trajectories.png",
            dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    recorded_epochs = list(trajectories.keys())
    if epochs is None:
        epochs = utils.sample_epochs(recorded_epochs, num_epochs)
        epochs = [epoch for epoch in epochs if epoch in recorded_epochs]

    exp = 0
    fig = plot_activity_and_weights_trajectories(data=(cfg, trajectories),
                                                 epochs=epochs, exp=exp, sess=0)
    if save_plots:
        fig.savefig(
            save_path /
            f"Exp_{cfg.logging.exp_id}_activity_and_weights (exp {exp}).png",
            dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    mode = "diff"
    fig = plot_init_weights_heatmaps(data=(cfg, experiments, trajectories),
                                     epochs=epochs, num_exps=5, mode=mode)
    if save_plots:
        fig.savefig(
            save_path / f"Exp_{cfg.logging.exp_id}_init_weights_heatmaps_{mode}.png")
    if show_plots:
        plt.show()
    plt.close(fig)

    mode = "abs"
    fig = plot_init_weights_traces(data=(cfg, experiments, trajectories), mode=mode)
    if save_plots:
        fig.savefig(
            save_path / f"Exp_{cfg.logging.exp_id}_init_weights_traces_{mode}.png")
    if show_plots:
        plt.show()
    plt.close(fig)

def plot_coeff_trajectories(path=None, data=None, params_table=None):
    if data:
        cfg, expdata = data
    elif path:
        cfg = utils.load_config(path)
        expdata = utils.load_expdata(path)
    else:
        raise ValueError("Either 'data' or 'path' must be provided.")

    exp_df = pd.DataFrame.from_dict(expdata)
    exp_id = int(cfg.logging.exp_id) if hasattr(cfg.logging, "exp_id") else None

    keymap = {"ff": "F", "rec": "R", "both": "B"}
    layers = list(cfg.plasticity.generation_plasticity.keys())
    prefixes = [keymap[layer_key] for layer_key in layers]
    if prefixes == ["F", "R", "B"]:
        prefixes = ["F", "R"]

    def has_reward(mask_array):
        mask_array = np.array(mask_array)
        return np.any(mask_array[:, :, :, 1:])

    masks = cfg.plasticity.coeff_masks
    use_all_81 = any(
        has_reward(masks[layer_key])
        for layer_key in masks
        if keymap.get(layer_key, "") in prefixes
    )

    if "epoch" in exp_df.columns:
        epochs = np.asarray(exp_df["epoch"])
    else:
        epochs = np.arange(len(exp_df))

    highlight = set()
    for layer_key in layers:
        prefix_char = keymap[layer_key]
        for gen_term in cfg.plasticity.generation_plasticity[layer_key]:
            highlight.add(
                f"{prefix_char}_{int(gen_term.pre)}{int(gen_term.post)}"
                f"{int(gen_term.weight)}{int(gen_term.reward)}"
            )

    coeff_columns = [
        col_name
        for col_name in exp_df.columns
        if isinstance(col_name, str)
        and len(col_name.split("_")) == 2
        and col_name[0] in "FRB"
        and col_name.split("_")[1].isdigit()
        and len(col_name.split("_")[1]) == 4
    ]

    columns_by_prefix = {
        prefix_char: [col for col in coeff_columns if col.startswith(prefix_char + "_")]
        for prefix_char in prefixes
    }

    def parse_key(col_name):
        s = col_name.split("_")[1]
        return tuple(map(int, list(s)))

    base_triples = []
    seen_triples = set()
    for col_name in sorted(coeff_columns, key=parse_key):
        a_idx, b_idx, w_power, r_idx = parse_key(col_name)
        triple = (a_idx, b_idx, w_power)
        if triple not in seen_triples:
            seen_triples.add(triple)
            base_triples.append(triple)

    cmap = plt.get_cmap("Set1")
    color_map = {}
    for weight_power in sorted({t[2] for t in base_triples}):
        triple_keys = [t for t in base_triples if t[2] == weight_power]
        matching_cols = [col for col in coeff_columns
                         if parse_key(col)[2] == weight_power]
        colors = (
            [cmap(0.5)]
            if len(triple_keys) == 1
            else [cmap(val) for val in np.linspace(0, 1, len(triple_keys))]
        )
        for triple_key, sample_color in zip(triple_keys, colors, strict=False):
            for col_name in matching_cols:
                if parse_key(col_name)[:3] == triple_key:
                    color_map[col_name] = sample_color

    def pretty_label(col_name):
        a_idx, b_idx, w_power, r_idx = parse_key(col_name)
        if a_idx == 0 and b_idx == 0 and w_power == 0 and r_idx == 0:
            return "1"
        parts = []
        for exponent, ch in ((a_idx, "x"), (b_idx, "y"), (w_power, "w"), (r_idx, "r")):
            if exponent == 0:
                continue
            parts.append(ch if exponent == 1 else f"{ch}^{{{exponent}}}")
        return "$" + "".join(parts) + "$"

    label_map = {col: pretty_label(col) for col in coeff_columns}
    linestyle_map = {0: "-", 1: "--", 2: ":"}

    metrics = ["PDE_F_neural", "PDE_T_neural", "PDE_W_neural"]
    has_eval = any(metric in exp_df.columns for metric in metrics)
    top_rows = 2 if has_eval else 1
    coeff_rows = 3 if use_all_81 else 1

    fig = plt.figure(
        figsize=(6 + 4 * len(prefixes), 3 + 2 * (top_rows + coeff_rows))
    )

    gs = fig.add_gridspec(
        top_rows + coeff_rows,
        len(prefixes),
        height_ratios=[1] * top_rows + [2] * coeff_rows,
    )
    fig.subplots_adjust(hspace=0.30, bottom=0.08)

    top_ax = fig.add_subplot(gs[0, :])
    top_ax.plot(
        epochs,
        exp_df["train_loss_median"],
        color="blue",
        label="train_loss_median",
    )
    if "test_loss_median" in exp_df.columns:
        top_ax.plot(
            epochs,
            exp_df["test_loss_median"],
            color="red",
            label="test_loss_median",
        )

    # exp_id and params as first line of the top subplot title
    if params_table and exp_id in params_table:
        params = params_table[exp_id]
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        top_ax.set_title(f"exp_{exp_id}: {params_str}\nTrain and Test Loss")
    else:
        top_ax.set_title(f"exp_{exp_id}\nTrain and Test Loss")

    top_ax.set_ylabel("Loss")
    top_ax.grid(True)
    top_ax.legend(loc="center right")

    eval_ax = None
    if has_eval:
        eval_ax = fig.add_subplot(gs[1, :])
        for metric, color in [
            ("PDE_F_neural", "blue"),
            ("PDE_T_neural", "purple"),
            ("PDE_W_neural", "red"),
        ]:
            if metric in exp_df.columns:
                eval_ax.plot(epochs, exp_df[metric], label=metric, color=color)
        eval_ax.set_title("Evaluation Metrics")
        eval_ax.set_ylabel("PDE")
        eval_ax.grid(True)
        eval_ax.legend(loc="center right", fontsize=8)

    start_row = top_rows
    coeff_axes = [
        [fig.add_subplot(gs[start_row + r, c]) for c in range(len(prefixes))]
        for r in range(coeff_rows)
    ]

    # increase vertical gap between coefficient rows only (fine-grained)
    if coeff_rows > 1:
        extra_gap = 0.045   # how much to push the 2nd row down relative to the first
        for row_idx in range(1, coeff_rows):
            shift = extra_gap * row_idx
            for ax in coeff_axes[row_idx]:
                pos = ax.get_position()
                # move this row downward by `shift` in figure coordinates
                ax.set_position([pos.x0, pos.y0 - shift, pos.width, pos.height])

    for col_idx, prefix_char in enumerate(prefixes):
        cols_for_prefix = columns_by_prefix[prefix_char]
        if not cols_for_prefix:
            continue
        if use_all_81:
            for row_idx in range(3):
                axis = coeff_axes[row_idx][col_idx]
                for col_name in sorted(cols_for_prefix, key=parse_key):
                    a_idx, b_idx, w_power, reward_idx = parse_key(col_name)
                    if reward_idx != row_idx:
                        continue
                    axis.plot(
                        epochs,
                        exp_df[col_name].values,
                        linestyle=linestyle_map.get(w_power, "-"),
                        color=color_map.get(col_name, "k"),
                        linewidth=(3 if col_name in highlight else 1.5),
                    )
                axis.grid(True)
        else:
            axis = coeff_axes[0][col_idx]
            for col_name in sorted(cols_for_prefix, key=parse_key):
                a_idx, b_idx, w_power, reward_idx = parse_key(col_name)
                axis.plot(
                    epochs,
                    exp_df[col_name].values,
                    linestyle=linestyle_map.get(w_power, "-"),
                    color=color_map.get(col_name, "k"),
                    linewidth=(3 if col_name in highlight else 1.5),
                )
            axis.grid(True)

    # Title each top coefficient subplot with its layer name (ff/rec/both)
    # ensure same y-limits across all coefficient axes
    if coeff_columns:
        global_min = min(exp_df[col].min() for col in coeff_columns)
        global_max = max(exp_df[col].max() for col in coeff_columns)
        if global_min == global_max:
            global_min -= 1e-6
            global_max += 1e-6
        pad = 0.05 * (global_max - global_min)
        y_min = global_min - pad
        y_max = global_max + pad
        for row_axes in coeff_axes:
            for axis in row_axes:
                axis.set_ylim(y_min, y_max)

    layers_for_prefixes = [layer for layer in layers if keymap[layer] in prefixes]
    for col_index, layer_name in enumerate(layers_for_prefixes):
        coeff_axes[0][col_index].set_title(f"Coefficients of {layer_name} layer")

    def proxy(col_name):
        a_idx, b_idx, w_power, r_idx = parse_key(col_name)
        color = color_map.get(col_name, "k")
        line_width = 1.5
        return Line2D(
            [0],
            [0],
            color=color,
            lw=line_width,
            linestyle=linestyle_map.get(w_power, "-"),
    )

    for row_index in range(3 if use_all_81 else 1):
        legend_entries = []
        for triple in base_triples:
            candidate_col = next(
                (col for col in coeff_columns
                 if parse_key(col)[:3] == triple and parse_key(col)[3] == row_index),
                None,
            )
            if candidate_col is None:
                candidate_col = next(
                    (col for col in coeff_columns if parse_key(col)[:3] == triple),
                    None,
                )
            if candidate_col is None:
                legend_entries.append((Line2D([0], [0], color="none"), ""))
                continue
            legend_entries.append((proxy(candidate_col), label_map[candidate_col]))
        if not legend_entries:
            continue
        handles, labels = zip(*legend_entries, strict=False)
        y = min(ax.get_position().y0 for ax in coeff_axes[row_index])
        if coeff_rows == 1:
            legend_anchor = (0.5, y - 0.13)
        else:
            legend_anchor = (0.5, y - 0.09)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=legend_anchor,
            ncol=9,
            fontsize=10,
            frameon=False,
            handlelength=2.4,
            markerscale=1.0,
            labelspacing=0.3,
            columnspacing=1.0,
            handletextpad=0.5,
        )

    return fig

def plot_init_weights_traces(path=None, data=None,
                             sample_size=10, num_exps=5, mode="abs"):
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

    if data:
        cfg, experiments, trajectories = data
    elif path:
        cfg = utils.load_config(path)
        experiments = utils.load_generated_experiments(path, cfg, mode="train")
        trajectories = utils.load_hdf5(path +
                                       f"Exp_{cfg.logging.exp_id}_trajectories.h5")
    else:
        raise ValueError("Either 'data' or 'path' must be provided.")

    num_exps = min(num_exps, len(experiments))

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

def plot_init_weights_heatmaps(path=None, data=None,
                               epochs=None, num_epochs=4, num_exps=5, mode="diff"):
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

    def plot_heatmap(row, col, layer, epoch, w_gen):
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

    if data:
        cfg, experiments, trajectories = data
    elif path:
        cfg = utils.load_config(path)
        experiments = utils.load_generated_experiments(path, cfg, mode="train")
        trajectories = utils.load_hdf5(path +
                                       f"Exp_{cfg.logging.exp_id}_trajectories.h5")
    else:
        raise ValueError("Either 'data' or 'path' must be provided.")

    num_exps = min(num_exps, len(experiments))

    recorded_epochs = list(trajectories.keys())
    if epochs is None:
        epochs = utils.sample_epochs(recorded_epochs, num_epochs)
        epochs = [epoch for epoch in epochs if epoch in recorded_epochs]
    num_epochs = len(epochs)
    if mode == 'abs':
        num_epochs += 1  # for ground-truth column
    epochs = np.array(epochs)

    layers = list(trajectories[epochs[0]][0]['init_weights'].keys())
    num_layers = len(layers)

    n_rows = num_exps * num_layers
    n_cols = len(epochs)
    if n_cols == 0:
        print("Epochs requested for plotting activity not found in trajectories.")
        return None

    fig, ax = plt.subplots(n_rows, n_cols+1, figsize=(2.5 * n_cols, 2.5 * n_rows),
                           gridspec_kw={'width_ratios': [1.0]*num_epochs + [0.03]},
                           layout='constrained')
    ax = ax.reshape(n_rows, n_cols+1)  # ensure 2D array even if n_cols=1
    for i in range(num_exps):
        exp = experiments[i]
        for j, layer in enumerate(layers):
            w_gen = exp.w_init_gen[layer]['w']
            # Set color scale limits based on ground-truth weights.
            # Allow training weights twice the range of gen weights
            v_min, v_max = np.min(w_gen)*2, np.max(w_gen)*2

            for k, epoch in enumerate(epochs):
                im = plot_heatmap(num_layers*i+j, k, layer, epoch, w_gen)
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

def plot_activity_and_weights_trajectories(path=None, data=None,
                                           epochs=None, num_epochs=4, exp=0, sess=0):
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
                    0, 1, cfg.experiment.num_place_neurons * wff.shape[-2]))
                colors_vis = plt.get_cmap('autumn')(np.linspace(
                    0, 1, (wff.shape[-1]-cfg.experiment.num_place_neurons-1)
                    * wff.shape[-2]))
                colors_v = plt.get_cmap("Greens")(np.linspace(
                    0, 1, cfg.experiment.num_velocity_neurons * wff.shape[-2]))
                colors = np.concatenate([colors_pl, colors_vis, colors_v])
            for trace, color in zip(traces_ff, colors, strict=True):
                ax[2, col_i].plot(trace, color=color, alpha=0.2)

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

    if data:
        cfg, trajectories = data
    elif path:
        cfg = utils.load_config(path)
        trajectories = utils.load_hdf5(path +
                                       f"Exp_{cfg.logging.exp_id}_trajectories.h5")
    else:
        raise ValueError("Either 'data' or 'path' must be provided.")

    recorded_epochs = list(trajectories.keys())
    if epochs is None:
        epochs = utils.sample_epochs(recorded_epochs, num_epochs)
        epochs = [epoch for epoch in epochs if epoch in recorded_epochs]

    n_cols = len(epochs)
    if n_cols == 0:
        print("Epochs requested for plotting activity not found in trajectories.")
        return None
    n_rows = 3
    if 'w_rec' in trajectories[epochs[0]][exp]['weights']:
        n_rows += 1

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows),
                           layout='tight')
    ax = ax.reshape(n_rows, n_cols)  # ensure 2D array even if n_cols=1
    for col_i, epoch in enumerate(epochs):
        plot_epoch_trajectories(epoch, col_i)

    return fig
