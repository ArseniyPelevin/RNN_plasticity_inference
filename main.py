import time

import jax
import numpy as np
import training
from omegaconf import OmegaConf

# coeff_mask = np.zeros((3, 3, 3, 3))
# coeff_mask[0:2, 0, 0, 0:2] = 1
coeff_mask = np.ones((3, 3, 3, 3))
coeff_mask[:, :, :, 1:] = 0  # Zero out reward coefficients

config = {
    "expid": 17, # For saving results and seeding random
    "use_experimental_data": False,
    "input_type": 'random',  # 'random' (Mehta et al., 2023) / 'task' (Sun et al., 2025)
    "fit_data": ["neural"],  # ["behavioral", "neural"]
    "trainable_init_weights": [],  # ['w_ff'], ['w_rec'], ['w_ff', 'w_rec'], []

# Experiment design
    "num_exp_train": 25,  # Number of experiments/trajectories/animals
    "num_exp_test": 5,

    # Below commented are real values as per CA1 recording article. Be modest for now
    # "mean_num_sessions": 9,  # Number of sessions/days per experiment
    # "sd_num_sessions": 3,  # Standard deviation of sessions/days per experiment
    # "mean_trials_per_session": 124,  # Number of trials/runs in each session/day
    # "sd_trials_per_session": 43,  # Standard deviation of trials in each session/day
    # "mean_trial_time": 29,  # s, including 2s teleportation
    # "std_trial_time": 10,  # s

    "mean_num_sessions": 1,  # Number of sessions/days per experiment/trajectory/animal
    "sd_num_sessions": 0,  # Standard deviation of sessions/days per animal
    "mean_trials_per_session": 1,  # Number of trials/runs in each session/day
    "sd_trials_per_session": 0,  # Standard deviation of trials in each session/day

    # For input_type 'random':
    "mean_steps_per_trial": 50,  # Number of sequential time steps in one trial/run
    "sd_steps_per_trial": 0,  # Standard deviation of steps in each trial/run
    # For input_type 'task':
    "dt": 1,  # s, time step of simulation
    "mean_trial_time": 29,  # s, including 2s teleportation
    "std_trial_time": 10,  # s
    "velocity_std": 2,  # cm/s
    "velocity_smoothing_window": 5,  # seconds
    "trial_distance": 230,  # cm, fixed

# Network architecture
    # For input_type 'task':
    "num_place_neurons": 20,
    "num_visual_neurons_per_type": 10,
    "num_velocity_neurons": 10,  # TODO how is it supposed to work?
    # For input_type 'random':
    "num_hidden_pre": 50,  # x, presynaptic neurons for plasticity layer

    "num_hidden_post": 50,  # y, postsynaptic neurons for plasticity layer
    "num_outputs": 1,  # m, binary decision (licking/not licking at this time step)
    "recurrent": True,  # Whether to include recurrent connections
    "plasticity_layers": ["ff"],  # ["ff", "rec"]
    # Fraction of postsynaptic neurons receiving FF input, for generation and training,
    # only effective if recurrent connections are present, otherwise 1
    "postsynaptic_input_sparsity_generation": 1,
    "postsynaptic_input_sparsity_training": 1,
    # Fraction of nonzero weights in feedforward layer, for generation and training,
    # of all postsynaptic neurons receiving FF input (postsynaptic_input_sparsity),
    # all presynaptic neurons are guaranteed to have some output
    "feedforward_sparsity_generation": 1,
    "feedforward_sparsity_training": 1,
    # Fraction of nonzero weights in recurrent layer, for generation and training,
    # all neurons receive some input (FF or rec, not counting self-connections)
    "recurrent_sparsity_generation": 1,
    "recurrent_sparsity_training": 1,

    "neural_recording_sparsity": 1,
    # TODO? output_sparsity?  # Fraction of postsynaptic neurons contributing to output

# Network dynamics
    "place_field_width_mean": 20,  # 70 cm - from article
    "place_field_width_std": 5,  # 50 cm - from article
    "place_field_amplitude_mean": 1.0,  # Units of firing rate
    "place_field_amplitude_std": 0.1,
    "place_field_center_jitter": 1.0,  # cm

    "presynaptic_firing_mean": 0,  # TODO rename into x
    "presynaptic_firing_std": 1,  # Input (before presynaptic) firing rates
    "presynaptic_noise_std": 0,  #0.05 # Noise added to presynaptic layer

    "feedforward_input_scale": 1,  # Scale of feedforward weights
    "recurrent_input_scale": 1,  # Scale of recurrent weights
    # TODO? Also different for generation and training?

    # Functional sparsity of plastic weights at initialization for generation only
    "init_weights_sparsity_generation": {'ff': 1, 'rec': 1},
    # Weight initialization mean for generation only
    "init_weights_mean_generation": {'ff': 0, 'rec': 0, 'out': 0},
    # Weight initialization std for generation and training. float or 'Xavier'
    "init_weights_std_generation": {'ff': 0.01, 'rec': 0.01, 'out': 0.01},
        # Could be used as prior?
    "init_weights_std_training": {'ff': 1, 'rec': 1, 'out': 1},

    "reward_scale": 0,
    "synaptic_weight_threshold": 6,  # Weights are normally in the range [-4, 4]

    "synapse_learning_rate": {'ff': 1, 'rec': 1},

    "measurement_noise_scale": 0,

# Plasticity
    "generation_plasticity": "1X1Y1W0R0-1X0Y2W1R0", # Oja's rule
    "generation_model": "volterra",  # "volterra", "mlp"
    "plasticity_model": "volterra",  # "volterra", "mlp"
    "plasticity_coeffs_init": "random",  # "zeros", "random"
    "plasticity_coeffs_init_scale": 1e-4,
    # Restrictions on trainable plasticity parameters
    "trainable_coeffs": int(np.sum(coeff_mask)),
    "coeff_mask": coeff_mask.tolist(),

# Training
    "num_epochs": 250,
    "learning_rate": 3e-3,
    "max_grad_norm": 0.2,

    "num_epochs_weights": 10,
    "learning_rate_weights": 1e-2,
    "max_grad_norm_weights": 1.0,
    "num_test_restarts": 5, # Random initializations per experiment to average loss over

    "regularization_type_theta": "none",  # "l1", "l2", "none"
    "regularization_scale_theta": 0,
    "regularization_type_weights": "none",  # "l1", "l2", "none"
    "regularization_scale_weights": 0,

# Logging
    "log_expdata": True,
    "log_interval": 10,
    "data_dir": "../../../../03_data/01_original_data/",
    "log_dir": "../../../../03_data/02_training_data/",
    "fig_dir": "../../../../05_figures/",

    "_return_weights_trajec": False,  # For debugging
}

def create_config():
    cfg = OmegaConf.create(config)
    cfg = validate_config(cfg)

    return cfg

def validate_config(cfg):
    # If no recurrent connectivity, recurrent is not plastic and not trainable
    if not cfg.recurrent and "rec" in cfg.plasticity_layers:
        cfg.plasticity_layers.remove("rec")
    if not cfg.recurrent and "w_rec" in cfg.trainable_init_weights:
        cfg.trainable_init_weights.remove("w_rec")

    cfg.init_weights_sparsity_generation = {
        k: float(v) for k, v in cfg.init_weights_sparsity_generation.items()
    }

    # Validate plasticity_model
    if cfg.plasticity_model not in ["volterra", "mlp"]:
        raise ValueError("Only 'volterra' and 'mlp' plasticity models are supported!")

    # Validate generation_model
    if cfg.generation_model not in ["volterra", "mlp"]:
        raise ValueError("Only 'volterra' and 'mlp' generation models are supported!")

    # Validate regularization_type
    if (cfg.regularization_type_theta.lower() not in ["l1", "l2", "none"] or
        cfg.regularization_type_weights.lower() not in ["l1", "l2", "none"]):
        raise ValueError(
            "Only 'l1', 'l2', and 'none' regularization types are supported!"
        )

    if type(cfg.fit_data) is str:
        cfg.fit_data = [cfg.fit_data]
    # Validate fit_data contains 'behavioral' or 'neural'
    if not ("behavioral" in cfg.fit_data or "neural" in cfg.fit_data):
        raise ValueError("fit_data must contain 'behavioral' or 'neural', or both!")

    if cfg.input_type == 'task':
        num_visual_types = 5  # Teleportation is not encoded, it is lack of input
        cfg.num_hidden_pre = (cfg.num_place_neurons
                              + num_visual_types * cfg.num_visual_neurons_per_type)
                              # + cfg.num_velocity_neurons)  # TODO

        cfg.mean_steps_per_trial = int(cfg.mean_trial_time / cfg.dt)
        cfg.sd_steps_per_trial = int(cfg.std_trial_time / cfg.dt)

    return cfg

def run_experiment(cfg, seed=None):
    cfg = validate_config(cfg)
    if seed is None:
        seed = cfg.expid
    cfg.seed = seed  # Save seed in config for logging
    key = jax.random.PRNGKey(seed)

    # Pass subkeys, so that adding more experiments doesn't affect earlier ones
    train_exp_key, test_exp_key, train_key, eval_key = jax.random.split(key, 4)

    experiments = training.generate_data(train_exp_key, cfg, mode='train')
    test_experiments = training.generate_data(test_exp_key, cfg, mode='test')

    time_start = time.time()
    expdata, _activation_trajs, _losses_and_r2s = (
        training.train(train_key, cfg, experiments, test_experiments))
    train_time = time.time() - time_start

    training.save_results(cfg, expdata, train_time)
    return _activation_trajs, _losses_and_r2s

if __name__ == "__main__":
    cfg = create_config()
    _activation_trajs, _losses_and_r2s = run_experiment(cfg)
