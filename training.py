
from functools import partial
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
    """ Generate all experiments/trajectories as instances of class Experiment. """

    if mode == "train":
        num_experiments = cfg.num_exp_train
        print(f"\nGenerating {num_experiments} trajectories")
    elif mode == "test":
        num_experiments = cfg.num_exp_test
        print(f"\nGenerating {num_experiments} trajectories")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Presplit keys for each experiment
    key, *experiment_keys = jax.random.split(key, num_experiments + 1)

    experiments = []
    for exp_i in range(num_experiments):
        exp = experiment.Experiment(experiment_keys[exp_i],
                                    exp_i, cfg,
                                    generation_coeff, generation_func,
                                    global_teacher_init_params,
                                    mode)
        experiments.append(exp)
        print(f"Generated experiment {exp_i} with {exp.mask.shape[0]} sessions")

    return key, experiments

def generate_data(key, cfg, mode="train"):
    # Generate model activity
    key, plasticity_key, params_key = jax.random.split(key, 3)
    #TODO add branching for experimental data
    generation_coeff, generation_func = synapse.init_plasticity(
        plasticity_key, cfg, mode="generation_model"
    )
    global_teacher_init_params = model.initialize_parameters(
        params_key,
        cfg["num_hidden_pre"], cfg["num_hidden_post"],
        cfg["init_params_scale"]
    )
    key, experiments = generate_experiments(
        key, cfg, generation_coeff, generation_func,
        global_teacher_init_params, mode,
    )

    return key, experiments

def initialize_training_params(key, cfg, experiments, new_params=None):
    key, init_plasticity_key, *init_params_keys = jax.random.split(
        key, len(experiments)+2)
    # Initialize parameters for training
    plasticity_coeffs, plasticity_func = synapse.init_plasticity(
        init_plasticity_key, cfg, mode="plasticity_model"
    )

    if new_params is not None:
        # TODO a temporary solution for same initial params
        global_student_init_params = new_params
    else:
        global_student_init_params = model.initialize_parameters(
            init_params_keys[0],
            cfg["num_hidden_pre"], cfg["num_hidden_post"],
            cfg["init_params_scale"]
        )
    # TODO use this for real training
    for exp in experiments:
        # Prepare different initial synaptic weights for each simulated experiment,
        # but for now use the same initialization for all students
        exp.new_init_params = global_student_init_params
        # exp.new_init_params = model.initialize_parameters(
        #         init_params_keys[exp.exp_i],
        #         cfg["num_hidden_pre"], cfg["num_hidden_post"]
        #         )

    return key, plasticity_coeffs, plasticity_func, experiments

def training_loop(key, cfg,
                  train_experiments, test_experiments,
                  loss_value_and_grad, optimizer, opt_state,
                  plasticity_coeffs, plasticity_func,
                  expdata):
    # Return simulation trajectory - for debugging purposes only,
    # set cfg._return_params_trajectory=True
    _activation_trajs = [[None for _ in range(len(train_experiments))]
                        for _ in range(cfg["num_epochs"] + 1)]
    for epoch in range(cfg["num_epochs"] + 1):  # +1 so that we have 250th epoch
        for exp in train_experiments:
            key, subkey = jax.random.split(key)
            (_loss, _activations), meta_grads = loss_value_and_grad(
                subkey,  # Pass subkey this time, because loss will not return key
                exp.input_params,
                exp.new_init_params,
                # Current plasticity coeffs, updated on each iteration:
                plasticity_coeffs,
                plasticity_func,  # Static within losses
                exp.data,
                exp.mask,
                cfg,  # Static within losses
            )
            _activation_trajs[epoch][exp.exp_i] = _activations
            updates, opt_state = optimizer.update(
                meta_grads, opt_state, plasticity_coeffs
            )
            plasticity_coeffs = optax.apply_updates(plasticity_coeffs, updates)

        if epoch % cfg.log_interval == 0:
            key, train_losses = evaluate_loss(
                key, cfg, plasticity_coeffs, plasticity_func, train_experiments
            )
            key, test_losses = evaluate_loss(
                key, cfg, plasticity_coeffs, plasticity_func, test_experiments
            )
            expdata = utils.print_and_log_training_info(
                cfg, expdata, plasticity_coeffs, epoch, train_losses, test_losses
            )
    return key, plasticity_coeffs, plasticity_func, expdata, _activation_trajs

def train(key, cfg, experiments, test_experiments):
    """Train the model with the given configuration and experiments.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary containing training parameters.
        experiments (list): List of experiments from class Experiment.
    """

    key, plasticity_coeffs, plasticity_func, experiments = (
        initialize_training_params(key, cfg, experiments)
    )

    key, _plasticity_coeffs, _plasticity_func, test_experiments = (
        initialize_training_params(key, cfg, test_experiments,
                                   #TODO a temporary solution for same init_params
                                   new_params=experiments[0].new_init_params)
    )

    # Return value (scalar) of the function (loss value)
    # and gradient wrt its parameter at argnum (plasticity_coeffs) - !Check argnums!
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=3, has_aux=True)

    # optimizer = optax.adam(learning_rate=cfg["learning_rate"])
    # Apply gradient clipping as in the article. Works on grad(coeffs), not weights!
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.2),
        optax.adam(learning_rate=cfg["learning_rate"]),
    )
    opt_state = optimizer.init(plasticity_coeffs)
    expdata = {}

    key, plasticity_coeffs, plasticity_func, expdata, _activation_trajs = training_loop(
        key, cfg,
        experiments, test_experiments,
        loss_value_and_grad, optimizer, opt_state,
        plasticity_coeffs, plasticity_func,
        expdata)

    return key, plasticity_coeffs, plasticity_func, expdata, _activation_trajs

def evaluate_loss(key, cfg,
                  plasticity_coeffs, plasticity_func,
                  experiments):
    eval_losses = []
    for exp in experiments:
        key, subkey = jax.random.split(key)
        loss, _activations = losses.loss(
            subkey,  # Pass subkey this time, because loss will not return key
            exp.input_params,
            exp.new_init_params,
            # Current plasticity coeffs, updated on each iteration:
            plasticity_coeffs,
            plasticity_func,  # Static within losses
            exp.data,
            exp.mask,
            cfg,  # Static within losses
        )

        eval_losses.append(loss)

    return key, jnp.array(eval_losses)

def evaluate_model(
    key,
    cfg,
    test_experiments,
    plasticity_coeffs,
    plasticity_func,
    expdata
):
    """Evaluate the trained model."""
    if cfg["num_exp_eval"] > 0:
        pass

        # r2_score, percent_deviance = model.evaluate(
        #     key,
        #     cfg,
        #     plasticity_coeff,
        #     plasticity_func,
        #     activations?
        # )
        # expdata["percent_deviance"] = percent_deviance
        # if not cfg["use_experimental_data"]:
        #     expdata["r2_weights"] = r2_score["weights"]
        #     expdata["r2_activity"] = r2_score["activity"]
    return key, expdata

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
