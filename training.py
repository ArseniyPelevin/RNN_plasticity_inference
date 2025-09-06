import experiment
import jax
import jax.numpy as jnp
import losses
import model
import numpy as np
import optax
import pandas as pd
import sklearn
import synapse
import utils


def generate_experiments(key, cfg,
                         generation_coeff, generation_func,
                         mode="train"):
    """ Generate all experiments/trajectories as instances of class Experiment. """

    if mode == "train":
        num_experiments = cfg.num_exp_train
    elif mode == "test":
        num_experiments = cfg.num_exp_test
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print(f"\nGenerating {num_experiments} {mode} trajectories")

    # Presplit keys for each experiment
    experiment_keys = jax.random.split(key, num_experiments)

    experiments = []
    for exp_i in range(num_experiments):
        exp = experiment.Experiment(experiment_keys[exp_i],
                                    exp_i, cfg,
                                    generation_coeff, generation_func,
                                    mode)
        experiments.append(exp)
        print(f"Generated {mode} experiment {exp_i} with {exp.mask.shape[0]} sessions")

    return experiments

def generate_data(key, cfg, mode="train"):
    # Generate model activity
    plasticity_key, experiments_key = jax.random.split(key, 2)
    #TODO add branching for experimental data
    generation_coeff, generation_func = synapse.init_plasticity(
        plasticity_key, cfg, mode="generation_model"
    )
    experiments = generate_experiments(
        experiments_key, cfg, generation_coeff, generation_func, mode,
    )

    return experiments

def initialize_training_params(key, cfg, experiments):
    init_params_keys = jax.random.split(key, len(experiments))

    for exp in experiments:
        # Different initial synaptic weights for each simulated experiment
        exp.new_init_params = model.initialize_parameters(
                init_params_keys[exp.exp_i],
                cfg["num_hidden_pre"], cfg["num_hidden_post"],
                cfg["init_params_scale"]
                )

    return experiments

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
                key, cfg, plasticity_coeffs, plasticity_func,
                train_experiments[:len(test_experiments)]
            )
            key, test_losses = evaluate_loss(
                key, cfg, plasticity_coeffs, plasticity_func,
                test_experiments
            )
            expdata = utils.print_and_log_training_info(
                cfg, expdata, plasticity_coeffs, epoch, train_losses, test_losses
            )
    return plasticity_coeffs, plasticity_func, expdata, _activation_trajs

def train(key, cfg, experiments, test_experiments):
    """Train the model with the given configuration and experiments.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary containing training parameters.
        experiments (list): List of experiments from class Experiment.
    """
    key, init_plasticity_key, train_key, test_key = jax.random.split(key, 4)

    # Initialize coefficients for training
    plasticity_coeffs, plasticity_func = synapse.init_plasticity(
        init_plasticity_key, cfg, mode="plasticity_model"
    )

    # Initialize new initial parameters for each train and test experiment
    experiments = initialize_training_params(train_key, cfg, experiments)
    test_experiments = initialize_training_params(test_key, cfg, test_experiments)

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

    plasticity_coeffs, plasticity_func, expdata, _activation_trajs = training_loop(
        key, cfg,
        experiments, test_experiments,
        loss_value_and_grad, optimizer, opt_state,
        plasticity_coeffs, plasticity_func,
        expdata)

    return plasticity_coeffs, plasticity_func, expdata, _activation_trajs

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
    if cfg["num_exp_test"] == 0:
        return expdata

    r2_score = {"weights": [], "activity": []}
    percent_deviance = []

    for exp in test_experiments:
        (model_params_key, null_params_key,
         model_key, null_key) = jax.random.split(key, 4)

        # Simulate model with learned plasticity coefficients (plasticity_coeff)
        new_model_init_params = model.initialize_parameters(
                model_params_key,
                cfg["num_hidden_pre"], cfg["num_hidden_post"],
                cfg["init_params_scale"]
                )
        simulated_model_data = model.simulate_trajectory(
            model_key,
            exp.input_params,
            new_model_init_params,
            plasticity_coeffs,  # Our current plasticity coefficients estimate
            plasticity_func,
            exp.data,
            exp.mask,
            cfg,
            mode='generation_test'
        )

        # Simulate model with zeros plasticity coefficients for null model
        plasticity_coeff_zeros, zero_plasticity_func = synapse.init_plasticity_volterra(
            key=None, init="zeros", scale=None
            )
        new_null_init_params = model.initialize_parameters(
                null_params_key,
                cfg["num_hidden_pre"], cfg["num_hidden_post"],
                cfg["init_params_scale"]
                )
        simulated_null_data = model.simulate_trajectory(
            null_key,
            exp.input_params,
            new_null_init_params,
            plasticity_coeff_zeros,
            zero_plasticity_func,
            exp.data,
            exp.mask,
            cfg,
            mode='generation_test'
        )

        percent_deviance.append(
            evaluate_percent_deviance(
                exp.data, simulated_model_data, simulated_null_data,
                exp.mask, mode=cfg.fit_data
            )
        )

        r2_score_exp = evaluate_r2_score(
                exp.mask,
                exp.data,
                exp.params_trajec,
                simulated_model_data,
                cfg
                )
        r2_score["activity"].append(r2_score_exp["activity"])
        if 'weights' in r2_score_exp:  # if not cfg.use_experimental_data
            r2_score["weights"].append(r2_score_exp["weights"])

    expdata["percent_deviance"] = np.median(percent_deviance)
    expdata["r2_activity"] = np.median(r2_score["activity"])
    print('Percent deviance explained:', expdata["percent_deviance"])
    print('R2 score (activity):', expdata["r2_activity"])
    if 'weights' in r2_score:
        expdata["r2_weights"] = np.median(r2_score["weights"])
        print('R2 score (weights):', expdata["r2_weights"])

    return expdata

def evaluate_percent_deviance(experimental_data,
                              simulated_model_data,
                              simulated_null_data,
                              mask,
                              mode='neural'):
    """ Percent deviance explained by model.
    Calculate neg log likelihoods between model and data.
    Calculate neg log likelihoods between zero model and data.
    Percent deviance explained: (D_null - D_model) / D_null

    Args:
        data (dict): Dictionary of experimental data.
        simulated_model_data: Simulated activations with learned coefficients.
        simulated_null_data: Simulated activations with zero coefficients.
        mask: Mask of valid steps.
        mode: ['neural', 'behavioral'].

    Returns:
        Percent deviance explained scalar
    """
    if mode == 'neural':
        exp_activations = experimental_data['ys']
        model_activations = simulated_model_data['ys']
        null_activations = simulated_null_data['ys']
    elif mode == 'behavioral':
        exp_activations = experimental_data['decisions']
        model_activations = simulated_model_data['outputs']
        null_activations = simulated_null_data['outputs']
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Flatten sessions and steps
    exp_activations = exp_activations.reshape(-1, *exp_activations.shape[2:])
    model_activations = model_activations.reshape(-1, *model_activations.shape[2:])
    null_activations = null_activations.reshape(-1, *null_activations.shape[2:])
    mask = mask.flatten()

    # Choose only valid steps
    exp_activations = exp_activations[mask]
    model_activations = model_activations[mask]
    null_activations = null_activations[mask]

    if mode == 'behavioral':
        model_deviance = utils.binary_deviance(model_activations, exp_activations)
        null_deviance = utils.binary_deviance(null_activations, exp_activations)
    elif mode == 'neural':
        model_deviance = utils.sse_deviance(model_activations, exp_activations)
        null_deviance = utils.sse_deviance(null_activations, exp_activations)

    percent_deviance = 100 * (null_deviance - model_deviance) / null_deviance
    return percent_deviance

def evaluate_r2_score(mask,
                      exp_data,
                      exp_param_traj,
                      model_data,
                      cfg
):
    """
    Functionality: Evaluates the R2 score for weights and activity.
    Args:
        mask (N_sessions, N_steps_per_session_max),
        exp_data: Dict of (N_sessions, N_steps_per_session_max, ...) tensors,
        exp_param_traj (
            (N_sessions, N_steps_per_session_max, N_hidden_pre, N_hidden_post),  # w
            (N_sessions, N_steps_per_session_max, N_hidden_post)  # b
            ),
        model_data: Tuple of (x, y, output, params) from model.simulate_trajectory,
        cfg

    Returns:
        Dict of R2 scores for activity (and weights).
    """
    r2_score = {}
    mask = mask.flatten()

    if cfg.fit_data == 'neural':
        exp_activations = exp_data['ys']
        model_activations = model_data['ys']
    elif cfg.fit_data == 'behavioral':
        exp_activations = exp_data['decisions']
        model_activations = model_data['outputs']

    # (N_sessions, N_steps_per_session_max, ...) -> (N_steps_per_experiment, ...)
    exp_activations = exp_activations.reshape(-1, *exp_activations.shape[2:])
    model_activations = model_activations.reshape(-1, *model_activations.shape[2:])

    # Choose valid steps
    exp_activations = exp_activations[mask]
    model_activations = model_activations[mask]

    # Convert to numpy for sklearn
    exp_activations = np.asarray(jax.device_get(exp_activations))
    model_activations = np.asarray(jax.device_get(model_activations))

    r2_score["activity"] = sklearn.metrics.r2_score(exp_activations,
                                                    model_activations)

    if not cfg.use_experimental_data:
        exp_weight_trajec = exp_param_traj[0]  # w
        model_weight_trajec = model_data['params'][0]  # w  # TODO? b?

        # (N_sessions, N_steps_per_session_max, N_hidden_pre, N_hidden_post) ->
        # (N_steps_per_experiment, N_hidden_pre * N_hidden_post)
        exp_weight_trajec = exp_weight_trajec.reshape(
            -1, np.prod(exp_weight_trajec.shape[2:]))
        model_weight_trajec = model_weight_trajec.reshape(
            -1, np.prod(model_weight_trajec.shape[2:]))

        # Choose valid steps
        exp_weight_trajec = exp_weight_trajec[mask]
        model_weight_trajec = model_weight_trajec[mask]

        # Convert to numpy for sklearn
        exp_weight_trajec = np.asarray(jax.device_get(exp_weight_trajec))
        model_weight_trajec = np.asarray(jax.device_get(model_weight_trajec))

        r2_score["weights"] = sklearn.metrics.r2_score(exp_weight_trajec,
                                                       model_weight_trajec)
    return r2_score

def save_results(cfg, expdata, train_time):
    """Save training logs and parameters."""
    df = pd.DataFrame.from_dict(expdata)
    df["train_time"] = train_time

    # Add configuration parameters to DataFrame
    for cfg_key, cfg_value in cfg.items():
        if isinstance(cfg_value, (float | int | str)):
            df[cfg_key] = cfg_value

    _logdata_path = utils.save_logs(cfg, df)
