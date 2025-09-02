
from typing import Any  # TODO get rid of

import experiment
import jax
import losses
import model
import optax
import pandas as pd
import synapse
import utils
from utils import sample_truncated_normal


def generate_experiments(key, cfg,
                         generation_coeff, generation_func,
                         global_teacher_init_params,
                         mode="train"):
    # Generate all experiments/trajectories
    if mode == "train":
        num_experiments = cfg.num_exp_train
        print(f"\nGenerating {num_experiments} trajectories")
    elif mode == "eval":
        num_experiments = cfg.num_exp_eval
        print(f"\nGenerating {num_experiments} trajectories")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    experiments = []
    # experiments_data = {}
    for exp_i in range(num_experiments):
        # Pick random number of sessions in this experiment given mean and std
        key, num_sessions = sample_truncated_normal(
            key, cfg["mean_num_sessions"], cfg["sd_num_sessions"]
        )
        exp = experiment.Experiment(exp_i, cfg,
                                    generation_coeff, generation_func,
                                    num_sessions,
                                    global_teacher_init_params)
        experiments.append(exp)
        # experiments_data[exp_i] = exp.data
        print(f"Generated experiment {exp_i} with {num_sessions} sessions")

    return key, experiments

def generate_data(key, cfg):
    # Generate model activity
    key, plasticity_key, params_key = jax.random.split(key, 3)
    #TODO add branching for experimental data
    generation_coeff, generation_func = synapse.init_plasticity(
        plasticity_key, cfg, mode="generation_model"
    )
    global_teacher_init_params = model.initialize_parameters(
        params_key,
        cfg["num_hidden_pre"], cfg["num_hidden_post"],
        cfg["initial_params_scale"]
    )
    key, experiments = generate_experiments(
        key, cfg, generation_coeff, generation_func,
        global_teacher_init_params, mode="train",
    )

    return key, experiments

def initialize_training_params(key, cfg, experiments):
    key, init_plasticity_key, *init_params_keys = jax.random.split(
        key, cfg["num_exp_train"]+2)
    # Initialize parameters for training
    plasticity_coeffs, plasticity_func = synapse.init_plasticity(
        init_plasticity_key, cfg, mode="plasticity_model"
    )

    global_student_init_params = model.initialize_parameters(
        init_params_keys[0],
        cfg["num_hidden_pre"], cfg["num_hidden_post"],
        cfg["initial_params_scale"]
    )
    # TODO use this for real training
    for exp in experiments:
        # Prepare different initial synaptic weights for each simulated experiment,
        # but for now use the same initialization for all students
        exp.new_initial_params = global_student_init_params
        # exp.new_initial_params = model.initialize_parameters(
        #         init_params_keys[exp.exp_i],
        #         cfg["num_hidden_pre"], cfg["num_hidden_post"]
        #         )

    return key, plasticity_coeffs, plasticity_func, experiments

def training_loop(key, cfg, experiments,
                  loss_value_and_grad, optimizer, opt_state,
                  plasticity_coeffs, plasticity_func,
                  expdata):
    for epoch in range(cfg["num_epochs"] + 1):
        for exp in experiments:
            key, subkey = jax.random.split(key)
            loss, meta_grads = loss_value_and_grad(
                subkey,  # Pass subkey this time, because loss will not return key
                exp.input_params,
                exp.initial_params,  # <--- TODO exp.new_initial_params,
                # Current plasticity coeffs, updated on each iteration:
                plasticity_coeffs,
                plasticity_func,  # Static within losses
                exp.data['inputs'],
                # exp.data['xs'],  # Not needed, will recompute from input_parameters
                exp.data['ys'],
                exp.data['decisions'],
                exp.data['rewards'],
                exp.data['expected_rewards'],
                exp.mask,
                cfg,  # Static within losses
            )
            updates, opt_state = optimizer.update(
                meta_grads, opt_state, plasticity_coeffs
            )
            plasticity_coeffs = optax.apply_updates(plasticity_coeffs, updates)

        if epoch % 10 == 0:
            expdata = utils.print_and_log_training_info(
                cfg, expdata, plasticity_coeffs, epoch, loss)
    return key, plasticity_coeffs, plasticity_func, expdata

def train(key, cfg, experiments):
    """Train the model with the given configuration and experiments.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary containing training parameters.
        experiments (list): List of experiments from class Experiment.
    """

    key, train_key = jax.random.split(key)
    key, plasticity_coeffs, plasticity_func, experiments = (
        initialize_training_params(train_key, cfg, experiments)
    )

    # Return value (scalar) of the function (loss value)
    # and gradient wrt its parameter at argnum (plasticity_coeffs)
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=3) # !Check argnums!

    optimizer = optax.adam(learning_rate=cfg["learning_rate"])
    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(0.2),  # Apply gradient clipping as in the article
    #     optax.adam(learning_rate=cfg["learning_rate"]),
    # )
    opt_state = optimizer.init(plasticity_coeffs)
    expdata = {}

    key, plasticity_coeffs, plasticity_func, expdata = training_loop(
        key, cfg, experiments,
        loss_value_and_grad, optimizer, opt_state,
        plasticity_coeffs, plasticity_func,
        expdata)

    return key, plasticity_coeffs, plasticity_func, expdata

def evaluate_model(
    key: jax.random.PRNGKey,
    cfg: dict[str, Any],
    plasticity_coeff: Any,
    plasticity_func: Any,
    expdata: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the trained model."""
    if cfg["num_exp_eval"] > 0:
        r2_score, percent_deviance = model.evaluate(
            key,
            cfg,
            plasticity_coeff,
            plasticity_func,
        )
        expdata["percent_deviance"] = percent_deviance
        if not cfg["use_experimental_data"]:
            expdata["r2_weights"] = r2_score["weights"]
            expdata["r2_activity"] = r2_score["activity"]
    return expdata

def save_results(
    cfg: dict[str, Any], expdata: dict[str, Any], train_time: float
) -> str:
    """Save training logs and parameters."""
    df = pd.DataFrame.from_dict(expdata)
    df["train_time"] = train_time

    # Add configuration parameters to DataFrame
    for cfg_key, cfg_value in cfg.items():
        if isinstance(cfg_value, (float | int | str)):
            df[cfg_key] = cfg_value

    _logdata_path = utils.save_logs(cfg, df)
