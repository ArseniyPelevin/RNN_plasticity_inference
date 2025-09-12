import time

import jax
import numpy as np
import training
from omegaconf import OmegaConf


def create_config():
    # coeff_mask = np.zeros((3, 3, 3, 3))
    # coeff_mask[0:2, 0, 0, 0:2] = 1
    coeff_mask = np.ones((3, 3, 3, 3))
    coeff_mask[:, :, :, 1:] = 0  # Zero out reward coefficients

    config = {
        "expid": 17, # For saving results and seeding random
        "use_experimental_data": False,
        "fit_data": "neural",  # ["behavioral", "neural"]
        "trainable_init_weights": [],  # ['w_ff'], ['w_rec'], ['w_ff', 'w_rec'], []

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
        "plasticity_layers": ["feedforward", "recurrent"],  # ["feedforward", "recurrent"]
        "postsynaptic_input_sparsity": 1,  # Fraction of posts. neurons receiving FF input,
            # only effective if recurrent connections are present, otherwise 1
        "feedforward_sparsity": 1,  # Fraction of nonzero weights in feedforward layer,
            # of all postsynaptic neurons receiving FF input (postsynaptic_input_sparsity),
            # all presynaptic neurons are guaranteed to have some output
        "recurrent_sparsity": 1,  # Fraction of nonzero weights in recurrent layer,
            # all neurons receive some input (FF or rec, not counting self-connections)
        "neural_recording_sparsity": 1,
        # TODO? output_sparsity?  # Fraction of postsynaptic neurons contributing to output

    # Network dynamics
        "input_weights_scale": 1,
        "presynaptic_firing_mean": 0,  # TODO rename into x
        "presynaptic_firing_std": 1,  # Input (before presynaptic) firing rates
        "presynaptic_noise_std": 0,  #0.05 # Noise added to presynaptic layer

        "feedforward_input_scale": 1,  # Scale of feedforward weights,
            # only if no feedforward plasticity
        "recurrent_input_scale": 1,  # Scale of recurrent weights,
            # only if no recurrent plasticity

        "init_weights_scale": {'ff': 0.01, 'rec': 0.01, 'out': 0.01},  # float or 'Xavier'

        "reward_scale": 0,
        "synaptic_weight_threshold": 6,  # Weights are normally in the range [-4, 4]

        "synapse_learning_rate": {'ff': 1, 'rec': 1},

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

    cfg = OmegaConf.create(config)
    cfg = validate_config(cfg)

    return cfg

def validate_config(cfg):
    # If no recurrent connectivity, recurrent is not plastic and not trainable
    if not cfg.recurrent and "recurrent" in cfg.plasticity_layers:
        cfg.plasticity_layers.remove("recurrent")
    if not cfg.recurrent and "w_rec" in cfg.trainable_init_weights:
        cfg.trainable_init_weights.remove("w_rec")

    if "recurrent" in cfg.trainable_init_weights:
        cfg.trainable_init_weights.remove("recurrent")
        cfg.trainable_init_weights.append("w_rec")
    if "feedforward" in cfg.trainable_init_weights:
        cfg.trainable_init_weights.remove("feedforward")
        cfg.trainable_init_weights.append("w_ff")

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

    # Validate fit_data contains 'behavior' or 'neural'
    if not ("behavior" in cfg.fit_data or "neural" in cfg.fit_data):
        raise ValueError("fit_data must contain 'behavior' or 'neural', or both!")

    return cfg

def run_experiment(cfg):
    cfg = validate_config(cfg)
    key = jax.random.PRNGKey(cfg["expid"])
    # Pass subkeys, so that adding more experiments doesn't affect earlier ones
    train_exp_key, test_exp_key, train_key, eval_key = jax.random.split(key, 4)

    experiments = training.generate_data(train_exp_key, cfg, mode='train')
    test_experiments = training.generate_data(test_exp_key, cfg, mode='test')

    time_start = time.time()
    learned_params, plasticity_func, expdata, _activation_trajs = (
        training.train(train_key, cfg, experiments, test_experiments))
    train_time = time.time() - time_start

    expdata = training.evaluate_model(eval_key, cfg,
                                      test_experiments,
                                      learned_params, plasticity_func,
                                      expdata)
    training.save_results(cfg, expdata, train_time)
    return _activation_trajs

if __name__ == "__main__":
    cfg = create_config()
    _activation_trajs = run_experiment(cfg)
