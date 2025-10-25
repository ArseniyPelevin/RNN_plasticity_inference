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

import jax
import jax.numpy as jnp

# jax.config.update('jax_log_compiles', True)
import main
import matplotlib.pyplot as plt
import numpy as np
import plotting_utils

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
         '/nscale_by_N_inputs': False, 'init_weights_std': 'ff=0.1, rec=0.1'},
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
    27: {'rule': "$xy-y^{2}w$", "trainable_w": "w_rec, w_ff, w_out", "input_type": "random", "n_coeffs": 81},
    28: {'rule': "$xy-y^{2}w$", "trainable_w": "w_rec, w_ff", "input_type": "random", "n_coeffs": 81, "seed": 27},
    29: {'rule': "$xy-y^{2}w$", "trainable_w": "w_rec, w_ff", "input_type": "random", "n_coeffs": 27, "recording_sparsity": 0.5},
    30: {'rule': "$xy-y^{2}w$", "trainable_w": "w_rec, w_ff", "input_type": "random", "n_coeffs": 27, "recording_sparsity": 0.25},
    31: {'rule': "$xy-y^{2}w$", "trainable_w": "w_rec, w_ff", "input_type": "random", "n_coeffs": 27, "recording_sparsity": 0.1,
         '\nN_pre': 20, 'N_post': 30,},
    32: {'rule': "$xy-y^{2}w$", "trainable_w": "w_rec, w_ff", "input_type": "random", "n_coeffs": 27, "recording_sparsity": 0.1,
         '\nN_pre': 20, 'N_post': 300,},

    64: {'plasticity_layers': ["ff"], 'trainable_w': "None",
         "\nuse_ff_bias": False, "recurrent_input_scale": 0, "input_noise_std": 0, "seed": 47,
         "\nrec_b_scale": "rec_w_scale"},
    65: {'plasticity_layers': ["ff", "rec"], 'trainable_w': "None",
         "\nuse_ff_bias": False, "recurrent_input_scale": 1, "input_noise_std": 0, "seed": 47,
         "\ninput_sparsity": 1},
    66: {'plasticity_layers': ["ff", "rec"], 'trainable_w': "None",
         "\nuse_ff_bias": False, "recurrent_input_scale": 1, "input_noise_std": 0, "seed": 47,
         "\ninput_sparsity_gen": 0.3, "input_sparsity_train": 1},
    67: {'plasticity_layers': ["ff", "rec"], 'trainable_w': "None",
         "\nuse_ff_bias": False, "recurrent_input_scale": 1, "input_noise_std": 0, "seed": 47,
         "\ninput_sparsity_gen": 0.3, "input_sparsity_train": 0.3},
    68: {'plasticity_layers': ["ff", "rec"], 'trainable_w': "None",
         "\nuse_ff_bias": False, "recurrent_input_scale": 1, "input_noise_std": 0, "seed": 47,
         "\ninput_sparsity_gen": 0.3, "input_sparsity_train": 0.3, "rec_bias": True},
    69: {'plasticity_layers': ["ff", "rec"], 'trainable_w': "None",
         "\nuse_ff_bias": False, "recurrent_input_scale": 0.5, "input_noise_std": 0, "seed": 47,
         "\ninput_sparsity_gen": 0.3, "input_sparsity_train": 0.3, "rec_bias": True},
    76: {'plasticity_layers': ["rec"], 'trainable_w': ["ff", "rec"],
         "\nnum_x": 50, "num_y": 100, "input_sparsity_gen": 0.3, "input_sparsity_train": 0.3,
         "\nrule": "$-xy + 0.6y - 0.1w$"},
    77: {"loss": "-H + 5 * edge", "L2": -0.1, "synaptic_weight_threshold": 10},
    78: {'plasticity_layers': ["rec"], 'trainable_w': ["ff", "rec"],
         "\nnum_x": 50, "num_y": 100, "input_sparsity_gen": 0.3, "input_sparsity_train": 0.3,
         "\nrule": "$-0.1xy^2 - 0.2xy - 0.2y^2 + 0.1w + 0.1$"},
    79: {'plasticity_layers': ["rec"], 'trainable_w': ["ff", "rec"],
         "\nnum_x": 10, "num_y": 20, "input_sparsity_gen": 0.3, "input_sparsity_train": 0.3,
         "\nrule": "$-0.1xy^2 - 0.2xy - 0.2y^2 + 0.1w + 0.1$"},
    80: {"loss": "-H + 5 * edge", "L2": -0.1, "synaptic_weight_threshold": 1000},
    81: {"loss": "5 * edge", "synaptic_weight_threshold": 1000},
    82: {"loss": "-H + 5 * edge", "synaptic_weight_threshold": 1000},
    83: {"loss": "-H + 7 * edge", "L2": -0.05, "synaptic_weight_threshold": 1000},
    84: {"trainable_weights": "none", "loss": "-H + 10 * edge", "L2": -0.01, "synaptic_weight_threshold": 1000},
    85: {"trainable_weights": "none", "loss": "-H + 9 * edge", "L2": -0.02, "synaptic_weight_threshold": 1000},
    86: {"trainable_weights": "none", "loss": "-H + 9 * edge", "L2": -0.01, "synaptic_learning_rate": 0.3},
    87: {'plasticity_layers': ["rec"], 'trainable_w': "none", "synaptic_learning_rate": 0.1, "input_type": "task",
         "\nnum_x": 20, "num_y": 30, "input_sparsity_gen": 0.3, "input_sparsity_train": 0.3,
         "\nrule": "$-0.15xy^2w - 0.2y^2w + 0.2xyw + 0.15yw - 0.03w$"},
    88: {'plasticity_layers': ["rec"], 'trainable_w': "none", "synaptic_learning_rate": 0.3, "input_type": "task",
         "\nnum_x": 29, "num_y": 30, "input_sparsity_gen": 0.5, "input_sparsity_train": 0.5, "ff_spars_gen": 1, "ff_spars_train": 1,
         "\nrule": "$-xy^2w + 0.5y^2$"},
    89: {"same_init_thetas": True, "same_init_weights": True, "same_init_connectivity": True},
    90: {"same_init_thetas": True, "same_init_weights": False, "same_init_connectivity": True},
    91: {"same_init_thetas": False, "same_init_weights": True, "same_init_connectivity": True},
    92: {"same_init_thetas": False, "same_init_weights": False, "same_init_connectivity": True},
    93: {"same_init_thetas": True, "same_init_weights": True, "same_init_connectivity": True, "same_input": True,},
    94: {"same_init_thetas": False, "same_init_weights": True, "same_init_connectivity": True, "same_input": True,
         "\ninput_sparsity": 1, "ff_sparsity": 1, "rec_sparsity": 0.3, "trials": 1},
    95: {"same_init_thetas": False, "same_init_weights": True, "same_init_connectivity": True, "same_input": True,
         "\ninput_sparsity": 1, "ff_sparsity": 1, "rec_sparsity": 0.3, "coeff_mask": False, "trials": 1},
    96: {"same_init_thetas": True, "same_init_weights": False, "same_init_connectivity": True, "same_input": True,
         "\ninput_sparsity": 1, "ff_sparsity": 1, "rec_sparsity": 0.3, "coeff_mask": False, "trials": 5},
    97: {"continued learning": "test"},
    98: {"same_init_thetas": True, "same_init_weights": True, "same_init_connectivity": True, "same_input": True,
         "\ninput_sparsity": 1, "ff_sparsity": 1, "rec_sparsity": 0.3, "coeff_mask": False},
    99: {"same_init_thetas": True, "same_init_weights": False, "same_init_connectivity": True, "same_input": True,
         "\ninput_sparsity": 1, "ff_sparsity": 1, "rec_sparsity": 0.3, "coeff_mask": False},
    100: {"same_init_thetas": False, "same_init_weights": True, "same_init_connectivity": True, "same_input": True,
         "\ninput_sparsity": 1, "ff_sparsity": 1, "rec_sparsity": 0.3, "coeff_mask": False},
    101: {"same_init_thetas": False, "same_init_weights": False, "same_init_connectivity": True, "same_input": True,
         "\ninput_sparsity": 1, "ff_sparsity": 1, "rec_sparsity": 0.3, "coeff_mask": False},
}

cfg = main.create_config()

# cfg.logging.exp_id = 97
# cfg.training.num_epochs = 100
# cfg.logging.log_interval = 1
path = r"C:\Users\pelevina\Desktop\Plasticity_inference_project\03_data\02_training_data\Exp_96/"
# params, expdata, trajectories, _losses_and_r2s = main.continue_experiment(path)
# print(f"{trajectories.keys()=}")

# cfg.training.same_init_thetas = True
# cfg.training.same_init_weights = False
# cfg.training.same_init_connectivity = True
# cfg.training.same_input = True

# path = None
# params, expdata, trajectories, _losses_and_r2s, path = main.run_new_experiment(cfg)
# if not path:  # When run to plot older experiment
#     path = cfg.logging.log_dir + f"Exp_{cfg.logging.exp_id}/"

plotting_utils.plot_init_weights_heatmaps(path, epochs=[350, 400, 800])

# plotting_utils.plot_experiment_results(path, behavioral_experiments_config_table, show_plots=True, save_plots=True)

# +
cfg = main.create_config()

print("EXPERIMENT 98")
cfg.training.same_init_thetas = True
cfg.training.same_init_weights = True
cfg.training.same_init_connectivity = True
cfg.logging.exp_id = 98
params, expdata, trajectories, _losses_and_r2s, path = main.run_new_experiment(cfg)
plotting_utils.plot_experiment_results(path, cfg, behavioral_experiments_config_table, show_plots=True)

print("EXPERIMENT 99")
cfg.training.same_init_thetas = True
cfg.training.same_init_weights = False
cfg.training.same_init_connectivity = True
cfg.logging.exp_id = 99
params, expdata, trajectories, _losses_and_r2s, path = main.run_new_experiment(cfg)
plotting_utils.plot_experiment_results(path, cfg, behavioral_experiments_config_table, show_plots=True)

print("EXPERIMENT 100")
cfg.training.same_init_thetas = False
cfg.training.same_init_weights = True
cfg.training.same_init_connectivity = True
cfg.logging.exp_id = 100
params, expdata, trajectories, _losses_and_r2s, path = main.run_new_experiment(cfg)
plotting_utils.plot_experiment_results(path, cfg, behavioral_experiments_config_table, show_plots=True)

print("EXPERIMENT 101")
cfg.training.same_init_thetas = False
cfg.training.same_init_weights = False
cfg.training.same_init_connectivity = True
cfg.logging.exp_id = 101
params, expdata, trajectories, _losses_and_r2s, path = main.run_new_experiment(cfg)
plotting_utils.plot_experiment_results(path, cfg, behavioral_experiments_config_table, show_plots=True)

# +
# inspect equinox module content
import dataclasses
from typing import Any

import equinox as eqx
import main


def inspect_eqx_module(module: eqx.Module, *, max_str_len: int = 200):
    """
    Pretty-print all properties (fields) of an eqx.Module.

    For each leaf we print:
      - path (dotted for module fields, [i] for sequences, ['k'] for dicts)
      - python type name
      - is_array (True if eqx thinks it's an array-like leaf)
      - traceable (True if it is an *inexact* array — floats — i.e. gradients possible)
      - shape (if array-like)

    Example usage:
        inspect_eqx_module(my_network)
    """
    def is_array(x: Any) -> bool:
        # eqx.is_array returns True for array leaves (including ints); keep it if available
        return eqx.is_array(x) if hasattr(eqx, "is_array") else isinstance(x, jnp.ndarray)

    def is_traceable_array(x: Any) -> bool:
        # inexact arrays (float) are traceable for grad
        if hasattr(eqx, "is_inexact_array"):
            return eqx.is_inexact_array(x)
        try:
            return isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.inexact)
        except Exception:
            return False

    def short(x: Any) -> str:
        s = repr(x)
        return s if len(s) <= max_str_len else s[: max_str_len - 3] + "..."

    def walk(obj: Any, path: str):
        # module -> iterate dataclass fields
        if isinstance(obj, eqx.Module):
            print(f"{path:60}  Module {type(obj).__name__}")
            for f in dataclasses.fields(type(obj)):
                name = f.name
                try:
                    val = getattr(obj, name)
                except Exception as e:
                    print(f"{path + '.' + name:60}  <error reading field: {e}>")
                    continue
                walk(val, path + "." + name)
            return

        # dict
        if isinstance(obj, dict):
            print(f"{path:60}  dict")
            for k, v in obj.items():
                walk(v, path + f"[{repr(k)}]")
            return

        # sequence
        if isinstance(obj, (list, tuple)):
            print(f"{path:60}  {type(obj).__name__} len={len(obj)}")
            for i, v in enumerate(obj):
                walk(v, path + f"[{i}]")
            return

        # array-like leaves
        if is_array(obj):
            # shape/dtype if possible
            shape = getattr(obj, "shape", None)
            dtype = getattr(obj, "dtype", None)
            traceable = is_traceable_array(obj)
            print(
                f"{path:60}  array  type={type(obj).__name__:<15} traceable={traceable!s:<5} shape={shape} dtype={dtype}"
            )
            return

        # other leaf
        print(f"{path:60}  {type(obj).__name__:<15}  value={short(obj)}")

    walk(module, type(module).__name__)

# print(type(network))
# print(type(batched_network))
# inspect_eqx_module(network)
# inspect_eqx_module(batched_network)
import experiment
import plasticity

key = jax.random.key(1912)
cfg = main.create_config()
cfg.logging.log_trajectories = False
train_experiments = experiment.generate_experiments(key, cfg, mode='train')
# params, expdata = main.run_new_experiment(cfg)
# plasticity = plasticity.initialize_plasticity(key, cfg.plasticity, 'training')
# exp = train_experiments[0]
# inspect_eqx_module(plasticity['both'])
inspect_eqx_module(train_experiments[0])

# +
# Generate and plot experiments with specific plasticity rules
import experiment

cfg = main.create_config()

def plot_xy(gen_experiments):
    fig, ax = plt.subplots(num_exp*2, 1, layout='tight', figsize=(8, 12))
    for i in range(num_exp):
        n_steps = gen_experiments[i].data['xs'][0].T.shape[1]
        imx = ax[2*i].imshow(gen_experiments[i].data['xs'][0].T, cmap='viridis',
                            interpolation='none', aspect='auto',
                            extent=[0, n_steps, 0, gen_experiments[i].data['xs'][0].T.shape[0]])
        ax[2*i].set_title(f'Exp {i}: X activity', fontsize=10)
        ax[2*i].set_ylabel('X neuron')
        plt.colorbar(imx, ax=ax[2*i])
        imy = ax[2*i+1].imshow(gen_experiments[i].data['ys'][0].T, cmap='viridis',
                            interpolation='none', aspect='auto',
                            extent=[0, n_steps, 0, gen_experiments[i].data['ys'][0].T.shape[0]])

        ax[2*i+1].set_title(f'Exp {i}: Y activity', fontsize=10)
        ax[2*i+1].set_ylabel('Y neuron')
        plt.colorbar(imy, ax=ax[2*i+1])
    ax[2*num_exp-1].set_xlabel('Step')
    plt.title(f'Generated X and Y activities with {f=}')
    plt.show()

def init_gen_experiments(f):
    cfg.plasticity.generation_plasticity = {
        'rec':
        # -0.07y2 - 0.08y + 0.06 - 0.6w: homeostasis-rate-dependent oscillations
        # [{'value': 0.6, 'pre': 0, 'post': 0, 'weight': 1, 'reward': 0},
        #  {'value': 0.06, 'pre': 0, 'post': 0, 'weight': 0, 'reward': 0},
        #  {'value': -0.08, 'pre': 0, 'post': 1, 'weight': 0, 'reward': 0},
        #  {'value': -0.07, 'pre': 0, 'post': 2, 'weight': 0, 'reward': 0}]

        # -0.1xy2 -0.2xy2 -0.2y +0.1w +0.1: uniform activity
        # [{'value': 0.1, 'pre': 0, 'post': 0, 'weight': 1, 'reward': 0},
        #  {'value': 0.1, 'pre': 0, 'post': 0, 'weight': 0, 'reward': 0},
        #  {'value': -0.2, 'pre': 0, 'post': 2, 'weight': 0, 'reward': 0},
        #  {'value': -0.2, 'pre': 1, 'post': 1, 'weight': 0, 'reward': 0},
        #  {'value': -0.1, 'pre': 1, 'post': 2, 'weight': 0, 'reward': 0}]

        # Exp 84, epoch 140 rule. Uniform activity and stable weights.
        # R_1210	-0.18946
        # R_0210	-0.17745
        # R_1110	0.169058
        # R_0110	0.163479
            # [{'value': -0.18946, 'pre': 1, 'post': 2, 'weight': 1, 'reward': 0},
            #  {'value': -0.17745, 'pre': 0, 'post': 2, 'weight': 1, 'reward': 0},
            #  {'value': 0.169058, 'pre': 1, 'post': 1, 'weight': 1, 'reward': 0},
            #  {'value': 0.163479, 'pre': 0, 'post': 1, 'weight': 1, 'reward': 0},

            # {'value': -0.03, 'pre': 0, 'post': 0, 'weight': 1, 'reward': 0}]
        # Modification and simplification of the above rule
            # [{'value': -0.15, 'pre': 1, 'post': 2, 'weight': 1, 'reward': 0},
            #  {'value': -0.2, 'pre': 0, 'post': 2, 'weight': 1, 'reward': 0},
            #  {'value': 0.2, 'pre': 1, 'post': 1, 'weight': 1, 'reward': 0},
            #  {'value': 0.15, 'pre': 0, 'post': 1, 'weight': 1, 'reward': 0},

            #  {'value': -0.03, 'pre': 0, 'post': 0, 'weight': 1, 'reward': 0}]

        # Exp 85-inspired. Irregular oscillations. Weights around +1. Exps 89-101
            # [{'value': -1, 'pre': 1, 'post': 2, 'weight': 1, 'reward': 0},
            # {'value': 0.5, 'pre': 0, 'post': 2, 'weight': 0, 'reward': 0},]
            # # For oscillations with constant input:
            # {'value': 0.2, 'pre': 0, 'post': 1, 'weight': 1, 'reward': 0}],

        # Exp 86 simplification: slow aperiodic waves
            # [{'value': -1, 'pre': 1, 'post': 2, 'weight': 1, 'reward': 0},
            #  {'value': 1, 'pre': 1, 'post': 1, 'weight': 1, 'reward': 0},
            #  {'value': 0.5, 'pre': 0, 'post': 1, 'weight': 1, 'reward': 0},
            #  {'value': -1, 'pre': 0, 'post': 2, 'weight': 1, 'reward': 0},
            #  {'value': 0.3, 'pre': 0, 'post': 1, 'weight': 0, 'reward': 0},

            #  {'value': -0.2, 'pre': 0, 'post': 0, 'weight': 1, 'reward': 0}]

        # Balanced Hebbian
            [
                {'value': (2*f - 1), 'pre': 1, 'post': 1, 'weight': 1, 'reward': 0},
                {'value': 1, 'pre': 1, 'post': 1, 'weight': 0, 'reward': 0},
                {'value': -f, 'pre': 1, 'post': 0, 'weight': 1, 'reward': 0},
                {'value': -f, 'pre': 0, 'post': 1, 'weight': 1, 'reward': 0}
            ]
    }

    key = jax.random.key(123)
    cfg.experiment.num_exp_train = num_exp
    # cfg.experiment.input_type = 'task'
    cfg.experiment.input_firing_mean = 0
    gen_experiments = experiment.generate_experiments(key, cfg, mode='test')

    return gen_experiments

num_exp = 5
for f in [0, 0.5, 1]:
    print(f"Generating experiments with f={f}")
    gen_experiments = init_gen_experiments(float(f))
    plot_xy(gen_experiments)
    plt.hist(gen_experiments[0].data['ys'][:, 0:100].flatten(), bins=30, label='Steps 0-100')
    plt.hist(gen_experiments[0].data['ys'][:, 100:200].flatten(), bins=30, label='Steps 100-200')
    plt.hist(gen_experiments[0].data['ys'][:, 200:300].flatten(), bins=30, label='Steps 200-300')
    plt.legend()
    plt.title(f'Y activity distribution for f={f}')
    plt.show()

fig, ax = plt.subplots(num_exp, 1, layout='tight', figsize=(6, 4))
for i in range(num_exp):
    w_rec = gen_experiments[i].weights_trajec['w_rec'][0]
    print(w_rec.shape)
    im = ax[i].plot(w_rec.reshape(w_rec.shape[0], -1), alpha=0.5)
    ax[i].set_title(f'Exp {i}: Recurrent weights', fontsize=10)
plt.show()

# +
# Chaotic network: phase portrait and neuron activities
import equinox as eqx
import jax
from matplotlib.collections import LineCollection


class Net(eqx.Module):
    J_ij: jnp.ndarray
    J: float
    N: int
    g: float
    dt: float

    def __init__(self, key, J, N, g, dt=0.01):
        k1, _ = jax.random.split(key)
        self.J = J; self.N = N; self.g = g; self.dt = dt
        self.J_ij = jax.random.normal(k1, (N, N)) * (J / jnp.sqrt(N))
        self.J_ij = self.J_ij - jnp.diag(jnp.diag(self.J_ij))  # zero diagonal

    @eqx.filter_jit
    def __call__(self, key, h):
        dh = -h + jnp.dot(self.J_ij, jnp.tanh(self.g * h))
        h += self.dt * dh
        return h

key = jax.random.PRNGKey(1234)
key, k1, k2 = jax.random.split(key, 3)
gJ = 2.3
model = Net(k1, J=np.sqrt(gJ), N=150, g=np.sqrt(gJ), dt=0.1)
h = jax.random.normal(k2, (model.N,))
hs = []
for t in range(100000):
    key, subkey = jax.random.split(key)
    h = model(subkey, h)
    hs.append(h)
hs = jnp.array(hs)

X = np.array(hs)                 # (T, N)
Xc = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
pcs = Xc @ Vt.T                  # projected data (T, N)
plt.figure(figsize=(6,6))
# plt.plot(pcs[:,0], pcs[:,1], alpha=0.7)
points = pcs[:, :2]
segments = np.stack([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='rainbow', norm=plt.Normalize(0, 1))
lc.set_array(np.linspace(0, 1, len(segments)))
# lc.set_linewidth(2)
ax = plt.gca()
ax.add_collection(lc)
ax.autoscale()
plt.colorbar(lc, label='time progression')

plt.scatter(pcs[0,0], pcs[0,1], color='green', label='start')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Phase portrait: PC1 vs PC2 (gJ={gJ}, N={model.N})")
plt.axis('equal')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 4))
plt.imshow(hs.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Neuron activity')
plt.xlabel('Time step')
plt.ylabel('Neuron index')
plt.title(f'RNN Neuron Activities Over Time (gJ={gJ}, N={model.N})')
plt.show()


# +
# Chaotic network: Map of largest Lyapunov exponent over (N, gJ)
import tqdm

Ns = [50, 100, 150, 200, 250, 300, 400, 500, 600, 750, 1000]
gJ_list = np.arange(1.0, 7.0, 0.5)
key = jax.random.PRNGKey(0)
steps = 2000
dt = 0.1

lyap_grid = np.zeros((len(Ns), len(gJ_list)))

tqdm_bar = tqdm.tqdm(total=len(Ns)*len(gJ_list))
for i,N in enumerate(Ns):
    for j,gJ in enumerate(gJ_list):
        key, k1, k2, k3 = jax.random.split(key, 4)
        model = Net(k1, J=np.sqrt(gJ), N=N, g=np.sqrt(gJ), dt=dt)
        map_step = lambda h: h + model.dt * (-h + jnp.dot(model.J_ij, jnp.tanh(model.g * h)))
        h = jax.random.normal(k2, (N,))
        v = jax.random.normal(k3, (N,)); v = v / jnp.linalg.norm(v)
        s = 0.0
        for _ in range(steps):
            h = map_step(h)
            _, jvp = jax.jvp(map_step, (h,), (v,))
            v = jvp
            nrm = jnp.linalg.norm(v) + 1e-16
            v = v / nrm
            s += jnp.log(nrm)
        lyap_grid[i, j] = float(s) / (steps * dt)
        tqdm_bar.update(1)
Ns_arr = np.array(Ns)
gJ_arr = np.array(gJ_list)
plt.figure(figsize=(7,5))
X, Y = np.meshgrid(Ns_arr, gJ_arr)                 # grid at sampled points
plt.pcolormesh(X, Y, lyap_grid.T, shading='auto')  # use .T to match shapes
c = plt.colorbar(); c.set_label('largest Lyapunov exponent')
plt.xlabel('N'); plt.ylabel('gJ'); plt.title('Lyapunov map')

plt.xticks(Ns_arr, rotation=45)   # ticks correspond exactly to sampled N
plt.yticks(gJ_arr)
plt.tight_layout()
plt.show()
# -

# Chaotic network: Plot distance between two close trajectories as a function of time
import jax
import jax.numpy as jnp
import numpy as np


def lyap_trace(model, h0, steps=2000, eps=1e-8):
    # copy of your key-handling but same subkeys for both trajectories
    key = jax.random.PRNGKey(0)
    h1 = h0 + eps * jax.random.normal(key, h0.shape)
    d = []
    k = key
    for t in range(steps):
        k, sub = jax.random.split(k)
        h0 = model(sub, h0)
        h1 = model(sub, h1)   # same subkey -> same noise
        d.append(float(jnp.linalg.norm(h1-h0)))
    d = np.array(d)
    return np.log(d + 1e-20)
h0 = jax.random.normal(jax.random.PRNGKey(7),(model.N,))
logd = lyap_trace(model, h0, steps=200000)
import matplotlib.pyplot as plt

plt.plot(logd); plt.xlabel("t"); plt.ylabel("log dist"); plt.show()

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
cfg.logging.exp_id = 180
cfg.num_x_neurons = 1000
cfg.num_y_neurons = 1000

cfg.num_epochs = 300

for steps in [5, 10, 15, 20, 50, 100, 150]:
    cfg.mean_steps_per_trial = steps
    cfg.logging.exp_id += 1
    _activation_trajs, _losses_and_r2s = main.run_new_experiment(cfg)

    param_table = {
        cfg.logging.exp_id: {
            'recurrent': True, 'plasticity': "ff, rec", 'train_w': "w_rec, w_ff",
           "\nN_in": 1000, "N_out": 1000, 'init_spars': 'ff: 0.5, rec: 0.5',
           '\ninit_w_mean': 'ff=-0.2, rec=0.3', 'init_w_std': 'ff=1, rec=1',
           '\nn_steps': steps}}

    fig = plot_coeff_trajectories(cfg.logging.exp_id, param_table,
                                use_all_81=False)
    fig.savefig(cfg.fig_dir + f"RNN_Exp{cfg.logging.exp_id} coeff trajectories.png",
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
    _activation_trajs = main.run_new_experiment(cfg)

    params_table = {cfg.logging.exp_id: {
        'trainable': str(cfg.trainable_init_weights),
        'inp_spar': cfg.input_sparsity_generation,
        'ff_spar': cfg.feedforward_sparsity_generation,
        'rec_spar': cfg.recurrent_sparsity_generation,
        'init_spar_gen': cfg.init_weights_sparsity_generation,
        }}

    fig = plot_coeff_trajectories(cfg.logging.exp_id, params_table,
                                use_all_81=False)
    fig.savefig(cfg.fig_dir + f"RNN_Exp{cfg.logging.exp_id} coeff trajectories.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

cfg.recurrent = True
cfg.plasticity_layers = ["ff", "rec"]
cfg.init_weights_std_training = {'ff': 0.01, 'rec': 0.01, 'out': 0}
last_logging.exp_id = 62
cfg.num_x_neurons = 100
cfg.num_y_neurons = 100
cfg.mean_steps_per_trial = 50
cfg.num_epochs = 300
i = 1

for trainable in [[], ["w_rec"], ["w_rec", "w_ff"]]:
    cfg.trainable_init_weights = trainable
    for input_spars in [0.3, 0.6, 1]:
        for ff_spars in [0.3, 0.6, 1]:
            for rec_spars in [0.3, 0.6, 1]:
                cfg.input_sparsity_generation = input_spars
                cfg.input_sparsity_training = input_spars
                cfg.feedforward_sparsity_generation = ff_spars
                cfg.feedforward_sparsity_training = ff_spars
                cfg.recurrent_sparsity_generation = rec_spars
                cfg.recurrent_sparsity_training = rec_spars

                cfg.logging.exp_id = last_logging.exp_id + i

                run(cfg)
                i += 1

    cfg.input_sparsity_generation = 1
    cfg.input_sparsity_training = 1
    cfg.feedforward_sparsity_generation = 1
    cfg.feedforward_sparsity_training = 1
    cfg.recurrent_sparsity_generation = 1
    cfg.recurrent_sparsity_training = 1

    for init_spar_ff in [0.3, 0.6, 1]:
        for init_spar_rec in [0.3, 0.6, 1]:
            cfg.init_weights_sparsity_generation = {'ff': init_spar_ff,
                                                    'rec': init_spar_rec}
            cfg.logging.exp_id = last_logging.exp_id + i
            run(cfg)
            i += 1

# cfg.init_weights_sparsity_generation = {'ff': 0.5, 'rec': 0.5}
# cfg.init_weights_mean_generation = {'ff': 2, 'rec': -1, 'out': 0}
# cfg.init_weights_std_generation = {'ff': 0.01, 'rec': 1, 'out': 0}
# cfg.init_weights_std_training = {'ff': 1, 'rec': 1, 'out': 0}




# +
# Run Exp10-16
print("\nEXPERIMENT 10")
cfg.logging.exp_id = 10
cfg.input_firing_std = 0.5
main.run_new_experiment()

print("\nEXPERIMENT 11")
cfg.logging.exp_id = 11
cfg.input_firing_std = 0.1
main.run_new_experiment()

cfg.input_firing_std = 1

print("\nEXPERIMENT 12")
cfg.logging.exp_id = 12
cfg.synaptic_learning_rate = 0.5
main.run_new_experiment()

print("\nEXPERIMENT 13")
cfg.logging.exp_id = 13
cfg.synaptic_learning_rate = 1
main.run_new_experiment()

cfg.synaptic_learning_rate = 0.1

print("\nEXPERIMENT 14")
cfg.logging.exp_id = 14
cfg.init_weights_std = 0.05
main.run_new_experiment()

print("\nEXPERIMENT 15")
cfg.logging.exp_id = 15
cfg.init_weights_std = 0.01
main.run_new_experiment()

cfg.init_weights_std = 0.1

print("\nEXPERIMENT 16")
cfg.logging.exp_id = 16
main.run_new_experiment()


# +
# Explore space of input-output layer sizes

cfg.num_exp_train = 25
cfg.input_noise_std = 0
cfg.input_firing_std = 1
cfg.synaptic_learning_rate = 1
cfg.init_weights_std = 0.01

for i, (N_in, N_out) in enumerate(list(itertools.product([10, 50, 100, 500, 1000],
                                      [10, 50, 100, 500, 1000]))):
    cfg.num_x_neurons = N_in
    cfg.num_y_neurons = N_out
    cfg.logging.exp_id = 50 + i
    main.run_new_experiment()
    params_dict = {cfg.logging.exp_id: {"N_in": N_in, "N_out": N_out}}
    fig = plot_coeff_trajectories(cfg.logging.exp_id, params_dict)
    fig.savefig(cfg.fig_dir + f"Exp{cfg.logging.exp_id} coeff trajectories.png",
            dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted Exp{cfg.logging.exp_id} with N_in={N_in}, N_out={N_out}")

# +
# Diagnose trajectories for NaN

_activation_trajs = main.run_new_experiment()
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
 return_model_activations) = main.run_new_experiment()
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
cfg.num_x_neurons = 50
cfg.num_y_neurons = 50
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
                cfg.logging.exp_id = exp_id
                print(f"\nEXPERIMENT {cfg.logging.exp_id}:")
                _activation_trajs = main.run_new_experiment()
                params_dict = {cfg.logging.exp_id: {"N_in": 50, "N_out": 50,
                                        "plasticity": "+".join(plasticity),
                                           "\ninp_spar": input_sparsity,
                                            "FF_spar": ff_sparsity,
                                            "rec_spar": rec_sparsity,
                                        #    "FF_scale": ff_scale,
                                        #    "rec_scale": rec_scale}}
                                                }}
                recurrent_experiments_config_table.update(params_dict)
                fig = plot_coeff_trajectories(cfg.logging.exp_id,
                                                recurrent_experiments_config_table,
                                                use_all_81=False)
                fig.savefig(cfg.fig_dir + f"RNN_Exp{cfg.logging.exp_id} coeff trajectories.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                i += 1
print(recurrent_experiments_config_table)


# +
# Plot xs and ys, optionally evolution of weights

# _activation_trajs, _model_activations, _null_activations = main.run_new_experiment()

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

ax_xs.set_xlim(-0.5, cfg.num_x_neurons)
ax_xs.set_ylim(cfg.mean_steps_per_trial, -0.5)
ax_ys.set_xlim(-0.5, cfg.num_y_neurons)
ax_ys.set_ylim(cfg.mean_steps_per_trial, -0.5)

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
key = jax.random.PRNGKey(cfg.logging.exp_id)
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
