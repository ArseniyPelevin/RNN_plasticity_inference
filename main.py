import time

import experiment
import jax
import numpy as np
import omegaconf
import pandas as pd
import training
import utils

# coeff_mask = np.zeros((3, 3, 3, 3))
# coeff_mask[0:2, 0, 0, 0:2] = 1
coeff_mask = np.ones((3, 3, 3, 3)).astype(np.bool)
coeff_mask[:, :, :, 1:] = False  # Zero out reward coefficients
coeff_masks = {'ff': coeff_mask, 'rec': coeff_mask}

config = {
    "experiment": {
        "use_experimental_data": False,
        "input_type": 'random',  # 'random' (Mehta 2023) / 'task' (Sun 2025)

        "num_exp_train": 5,  # Number of experiments/trajectories/animals
        "num_exp_test": 5,

        # Commented below are real values as per CA1 recording article (Sun 2025)
        # "mean_num_sessions": 9,  # Number of sessions per experiment
        # "std_num_sessions": 3,  # Standard deviation of sessions per experiment
        # "mean_trials_per_session": 124,  # Number of trials per session
        # "std_trials_per_session": 43,  # Standard deviation of trials per session
        # "mean_trial_time": 29,  # s, including 2s teleportation
        # "std_trial_time": 10,  # s

        "mean_num_sessions": 1,  # Number of sessions/days per experiment
        "std_num_sessions": 0,  # Standard deviation of sessions per animal
        "mean_trials_per_session": 1,  # Number of trials/runs per session
        "std_trials_per_session": 0,  # Standard deviation of trials per session

        # For input_type 'random':
        # (Overwritten as mean_trial_time/dt if input_type is 'task')
        "mean_steps_per_trial": 50,  # Number of sequential time steps in one trial
        "std_steps_per_trial": 0,  # Standard deviation of steps per trial

        # For input_type 'task':
        "dt": 1,  # s, time step of simulation
        "mean_trial_time": 29,  # s, including 2s teleportation
        "std_trial_time": 10,  # s
        "velocity_std": 2,  # cm/s
        "velocity_smoothing_window": 5,  # seconds
        "trial_distance": 230,  # cm, fixed

        "num_place_neurons": 10,
        "num_visual_neurons_per_type": 5,
        "num_velocity_neurons": 1,

        "place_field_width_mean": 20,  # 70 cm - from article
        "place_field_width_std": 5,  # 50 cm - from article
        "place_field_amplitude_mean": 1.0,  # Units of firing rate
        "place_field_amplitude_std": 0.1,
        "place_field_center_jitter": 1.0,  # cm

        "input_firing_mean": 0,  # Mean firing rate of random input
        "input_firing_std": 1,  # Standard deviation of random input
    },

    "network": {
    # Network architecture
        # For input_type 'task', num_x_neurons is set automatically
        "num_x_neurons": 50,  # x, presynaptic neurons for feedforward layer
        "num_y_neurons": 100,  # y, neurons of recurrent layer
        "num_outputs": 1,

        "plasticity_layers": ["ff", "rec"],  # ["ff", "rec"]
        # Fraction of Y neurons receiving FF input, for generation and training,
        # only effective if recurrent connections are present, otherwise 1
        "input_sparsity": {"generation": 1, "training": 1},
        # Fraction of nonzero weights in feedforward layer, for generation and training,
        # of all Y neurons receiving FF input (input_sparsity),
        # all X neurons are guaranteed to have some output
        "feedforward_sparsity": {"generation": 1, "training": 1},
        # Fraction of nonzero weights in recurrent layer, for generation and training,
        # all Y neurons receive some input (FF or rec, not counting self-connections)
        "recurrent_sparsity": {"generation": 1, "training": 1},

        "neural_recording_sparsity": 1,
        # TODO? output_sparsity?  # Fraction of Y neurons contributing to output

    # Network dynamics
        "input_noise_std": 0,  #0.05 # Noise added to input layer

        "feedforward_input_scale": 1,  # Scale of feedforward weights
        "recurrent_input_scale": 1,  # Scale of recurrent weights
        # TODO? Also different for generation and training?

        # Weight initialization std for generation and training
        "init_weights_std": {"generation": {'ff': 0.01, 'rec': 0.01, 'out': 0.01},
                             "training": {'ff': 1, 'rec': 1, 'out': 1}},

        "reward_scale": 0,
        "synaptic_weight_threshold": 6,  # Weights are normally in the range [-4, 4]
        "min_lick_probability": 0.05,  # To encourage exploration, only in reinforcement

        "homeostasis_rate": 0.1,  # Rate of bias adaptation to mean post activity

        "measurement_noise_scale": 0,
    },

    "training": {
        "fit_data": ["neural"],  # ["behavioral", "neural"]

        # Only affects training, generation depends on generation_plasticity
        "trainable_thetas": "different",  # "same", "different"
        "trainable_init_weights": ['ff', 'rec', 'out'],

        "num_epochs": 250,
        "learning_rate": 3e-3,
        "max_grad_norm": 0.2,

        # Learning init weights in evaluation with fixed theta
        "num_epochs_weights": 10,
        "learning_rate_weights": 1e-2,
        "max_grad_norm_weights": 1.0,

        # Regularization
        "reg_types_theta": "none",  # "l1", "l2", "none"
        "reg_scales_theta": 0,
        "reg_types_weights": "none",  # "l1", "l2", "none"
        "reg_scales_weights": 0,

        "lick_cost": 0.1,  # Cost of each lick in reward-based tasks
    },

    # All plasticity parameters can be different for ff and rec layers if set as dict.
    # If set as single value, the same will be used for all plastic layers.
    "plasticity": {
        # Per-plastic-layer dict or str, in latter case use same for all plastic layers
        "generation_plasticity": {
            # 1xy - 1y2w
            'ff': [{'value': 1, 'pre': 1, 'post': 1, 'weight': 0, 'reward': 0},
                {'value': -1, 'pre': 0, 'post': 2, 'weight': 1, 'reward': 0}],
            'rec': [{'value': 1, 'pre': 1, 'post': 1, 'weight': 0, 'reward': 0},
                {'value': -1, 'pre': 0, 'post': 2, 'weight': 1, 'reward': 0}]
        },
        "generation_models": "volterra",  # "volterra" / "mlp"
        "plasticity_models": "volterra",  # "volterra" / "mlp"
        "plasticity_coeffs_init_scale": 1e-4,
        # Restrictions on trainable plasticity parameters
        "num_trainable_coeffs": {layer: int(np.sum(mask))
                            for layer, mask in coeff_masks.items()},
        # Per-plastic-layer masks for plasticity coefficients
        "coeff_masks": {k: v.tolist() for k, v in coeff_masks.items()},

        "synapse_learning_rate": 1,
    },

    "logging": {
        "exp_id": 17, # For saving results and seeding random
        "log_interval": 10,

        "do_evaluation": True,
        "log_config": True,
        "log_generated_experiments": False,
        "log_trajectories": True,
        "log_expdata": True,
        "log_final_params": True,

        "data_dir": "../../../../03_data/01_original_data/",
        "log_dir": "../../../../03_data/02_training_data/",
        "fig_dir": "../../../../05_figures/"
    }
}

def create_config():
    cfg = omegaconf.OmegaConf.create(config)

    return cfg

def validate_config(cfg):
    for layer in cfg.network.plasticity_layers:
        if layer not in ["ff", "rec"]:
            raise ValueError("Only 'ff' and 'rec' plastic layers are supported!")

    if type(cfg.training.fit_data) is str:
        cfg.training.fit_data = [cfg.training.fit_data]
    # Validate fit_data contains 'neural' and/or 'behavioral' or 'reinforcement'
    neural_or_behavioral = ("behavioral" in cfg.training.fit_data or
                            "neural" in cfg.training.fit_data)
    if (not neural_or_behavioral and "reinforcement" not in cfg.training.fit_data
        or neural_or_behavioral and "reinforcement" in cfg.training.fit_data):
        raise ValueError("fit_data must contain 'behavioral' and/or 'neural',"
                         " or 'reinforcement'")

    if 'reinforcement' not in cfg.training.fit_data:
        cfg.network.min_lick_probability = 0.0  # No prior strategy for fitting

    if cfg.experiment.input_type == 'task':
        num_visual_types = 6  # grey, ind1, ind2, reward1, reward2, teleportation
        cfg.network.num_x_neurons = (
            cfg.experiment.num_place_neurons
            + num_visual_types * cfg.experiment.num_visual_neurons_per_type
            + cfg.experiment.num_velocity_neurons
            )
        cfg.experiment.mean_steps_per_trial = int(cfg.experiment.mean_trial_time /
                                                  cfg.experiment.dt)
        cfg.experiment.std_steps_per_trial = int(cfg.experiment.std_trial_time /
                                                 cfg.experiment.dt)

    def convert_to_float(cfg):
        """Recursively convert int leaves of config to float."""
        if isinstance(cfg, omegaconf.dictconfig.DictConfig):
            cfg = {k: convert_to_float(v) for k, v in cfg.items()}
        elif isinstance(cfg, list):
            cfg = [convert_to_float(v) for v in cfg]
        elif isinstance(cfg, int):
            cfg = float(cfg)
        return cfg

    for section in config:
        for param in config[section]:
            if "sparsity" in param or "scale" in param or "std" in param:
                cfg[section][param] = convert_to_float(cfg[section][param])

    for param in cfg.plasticity:
        # If set as single value, convert to per-plastic-layer dict using that value
        if type(cfg.plasticity[param]) is not omegaconf.dictconfig.DictConfig:
            cfg.plasticity[param] = dict.fromkeys(cfg.network.plasticity_layers,
                                                  cfg.plasticity[param])
        # Ensure plasticity parameters given for all plastic layers
        for layer in cfg.network.plasticity_layers:
            if layer not in cfg.plasticity[param]:
                raise ValueError(f"Parameter {param} must be given for all plastic "
                                 f"layers {cfg.network.plasticity_layers}!")
        # Remove plasticity parameters for non-plastic layers
        for layer in cfg.plasticity[param]:
            if layer not in cfg.network.plasticity_layers:
                cfg.plasticity[param] = {k: v for k, v in cfg.plasticity[param].items()
                                         if k != layer}

        # If trainable_thetas='same' and layer values are same, collapse to layer 'both'
        if cfg.training.trainable_thetas == 'same' and len(cfg.plasticity[param]) > 1:
            # Ensure in 'same' mode layer values are the same
            if cfg.plasticity[param]['ff'] == cfg.plasticity[param]['rec']:
                cfg.plasticity[param] = {'both': cfg.plasticity[param]['ff']}
            else:
                raise ValueError(f"When trainable_thetas is 'same', "
                                 f"all plastic layers must have the same {param}!")

    # Ensure generation_plasticity for each plastic layer is a list of dicts
    for layer in cfg.plasticity.generation_plasticity:
        if (type(cfg.plasticity.generation_plasticity[layer])
            is not omegaconf.listconfig.ListConfig):
            cfg.plasticity.generation_plasticity[layer] = [
                cfg.plasticity.generation_plasticity[layer]]
        for coeff_dict in cfg.plasticity.generation_plasticity[layer]:
            if type(coeff_dict) is not omegaconf.dictconfig.DictConfig:
                raise ValueError(f"Each element of generation_plasticity for layer "
                                 f"{layer} must be a dict of coefficient values!")

    # Validate plasticity parameters
    for layer in cfg.plasticity.plasticity_models:
        # Validate generation_model
        if cfg.plasticity.generation_models[layer] not in ["volterra", "mlp"]:
            raise ValueError(
                "Generation model should be 'volterra' or 'mlp'!")
        # Validate plasticity_model
        if cfg.plasticity.plasticity_models[layer] not in ["volterra", "mlp"]:
            raise ValueError(
                "Plasticity model should be 'volterra' or 'mlp'!")

    # Validate regularization_type for theta
    if cfg.training.reg_types_theta.lower() not in ["l1", "l2", "none"]:
        raise ValueError(
            "Regularization type for theta should be 'l1', 'l2', or 'none'!")
    # Validate regularization_type for weights
    if cfg.training.reg_types_weights.lower() not in ["l1", "l2", "none"]:
            raise ValueError(
                "Regularization type for weights should be 'l1', 'l2', or 'none'!")

    return cfg

def run_experiment(cfg, seed=None):
    cfg = validate_config(cfg)
    if seed is None:
        seed = cfg.logging.exp_id
    cfg.experiment.seed = seed  # Save seed in config for logging
    key = jax.random.PRNGKey(seed)

    # Pass subkeys, so that adding more experiments doesn't affect earlier ones
    exp_key1, exp_key2, train_key = jax.random.split(key, 3)

    train_experiments = experiment.generate_experiments(exp_key1, cfg,
                                                        mode='train')
    test_experiments = experiment.generate_experiments(exp_key2, cfg, mode='test')

    time_start = time.time()
    # params, expdata, _activation_trajs, _losses_and_r2s = (
    params, expdata = (
        training.train(train_key, cfg, train_experiments, test_experiments))
    train_time = time.time() - time_start
    print(f"\nTraining time: {train_time:.1f} seconds")

    try:
        save_results(cfg, params, expdata, train_time)#, _activation_trajs)
    except Exception as e:
        print(f"Error saving results: {e}")

    return params, expdata#, _activation_trajs, _losses_and_r2s

def save_results(cfg, params, expdata, train_time, activation_trajs):
    """Save training logs and parameters."""

    # Save final parameters
    if cfg.log_final_params:
        export_dict = {'params': params,
                    #   'trajectories': {str(e): tr
                    #                    for e, tr in enumerate(activation_trajs)}
                      }
        utils.save_nested_hdf5(export_dict,
                               cfg.log_dir +
                               f"exp_{cfg.logging.exp_id}_final_params.h5")

    df = pd.DataFrame.from_dict(expdata)
    df["train_time"] = train_time

    # Add configuration parameters to DataFrame
    for cfg_key, cfg_value in cfg.items():
        if isinstance(cfg_value, (float | int | str)):
            df[cfg_key] = cfg_value
        elif isinstance(cfg_value, omegaconf.dictconfig.DictConfig):
            df[cfg_key] = ', '.join(f"{k}: {v}" for k, v in cfg_value.items())
        elif isinstance(cfg_value, omegaconf.listconfig.ListConfig):
            df[cfg_key] = ', '.join(str(v) for v in cfg_value)

    # Save expdata as .csv
    if cfg.log_expdata:
        _logdata_path = utils.save_logs(cfg, df)

if __name__ == "__main__":
    cfg = create_config()
    params, expdata, _activation_trajs, _losses_and_r2s = run_experiment(cfg)
