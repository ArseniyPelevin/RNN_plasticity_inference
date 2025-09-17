import evaluation
import experiment
import jax
import jax.numpy as jnp
import losses
import model
import numpy as np
import omegaconf
import optax
import pandas as pd
import sklearn
import synapse
import utils


def generate_experiments(key, cfg,
                         generation_theta, generation_func,
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
                                    generation_theta, generation_func,
                                    mode)
        experiments.append(exp)
        print(f"Generated {mode} experiment {exp_i}",
              f"with {exp.step_mask.shape[0]} sessions")

    return experiments

def generate_data(key, cfg, mode="train"):
    # Generate model activity
    plasticity_key, experiments_key = jax.random.split(key, 2)
    #TODO add branching for experimental data
    generation_theta, generation_func = synapse.init_plasticity(
        plasticity_key, cfg, mode="generation_model"
    )
    experiments = generate_experiments(
        experiments_key, cfg, generation_theta, generation_func, mode,
    )

    return experiments

def initialize_simulation_weights(key, cfg, experiments, n_restarts=1):
    """ Initialize new initial weights of all layers for each experiment.
    Store fixed - as init_fixed_weights property of each experiment instance,
    trainable - as per-restart list of per-layer dicts of per-exp arrays.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary.
        experiments (list): List of experiments from class Experiment.
        n_restarts (int): Number of random trainable weights initializations
            per experiment. Default is 1 for training, can be >1 for evaluation.

    Returns:
        experiments (list): Updated experiments with init_fixed_weights property.
        init_trainable_weights (list): Per-restart list of per-layer dicts
            of per-exp arrays of randomly initialized weights.
    """
    all_layers = ['w_ff', 'w_out']
    if cfg.recurrent:
        all_layers.append('w_rec')

    fixed_layers = [layer for layer in all_layers
                    if layer not in cfg.trainable_init_weights]
    trainable_layers = cfg.trainable_init_weights

    # Initialize trainable weights
    init_trainable_weights = [{layer: [] for layer in trainable_layers}
                              for _ in range(n_restarts)]
    # Different initial weights for each restart of evaluation on test experiments
    for start in range(n_restarts):
        for layer in trainable_layers:
            layer_weights = []
            # Different initial synaptic weights for each simulated experiment
            for _exp in range(len(experiments)):
                key, subkey = jax.random.split(key)
                layer_weights.append(model.initialize_weights(
                    subkey, cfg, cfg.init_weights_std_training, layers=layer)[layer])
            init_trainable_weights[start][layer] = jnp.array(layer_weights)

    # Initialize fixed weights
    for exp in experiments:
        exp.init_fixed_weights = {}
        for layer in fixed_layers:
            key, subkey = jax.random.split(key)
            exp.init_fixed_weights[layer] = model.initialize_weights(
                subkey, cfg, cfg.init_weights_std_training, layers=layer)[layer]

    return experiments, init_trainable_weights

def train(key, cfg, train_experiments, test_experiments):
    """ Initialize values and functions, start training loop.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary.
        train_experiments (list): List of training experiments from class Experiment.
        test_experiments (list): List of test experiments from class Experiment.
    """
    key, init_plasticity_key, train_key, test_key = jax.random.split(key, 4)

    # Initialize plasticity coefficients for training
    init_theta, plasticity_func = synapse.init_plasticity(
        init_plasticity_key, cfg, mode="plasticity_model"
    )

    # Initialize weights of all layers for each train and test experiment. Store them:
    # fixed - as init_fixed_weights property of each experiment instance,
    # trainable - as (per-restart list of) per-layer dicts of per-exp arrays
    train_experiments, init_trainable_weights_train = initialize_simulation_weights(
        train_key, cfg, train_experiments)
    test_experiments, init_trainable_weights_test = initialize_simulation_weights(
        test_key, cfg, test_experiments, n_restarts=cfg.num_test_restarts
    )

    params = {'theta': init_theta,
              'weights': init_trainable_weights_train[0]}  # n_restarts=1 for training

    # Return value (scalar) of the function (loss value) and gradient wrt its
    # parameters at argnums (theta and init_weights) - !Check argnums!
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=(5, 6), has_aux=True)

    # optimizer = optax.adam(learning_rate=cfg["learning_rate"])
    # Apply gradient clipping as in the article
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(learning_rate=cfg["learning_rate"]),
    )

    opt_state = optimizer.init(params)

    # Return simulation trajectory - for debugging purposes only,
    # set cfg._return_weights_trajectory=True
    _activation_trajs = [[None for _ in range(len(train_experiments))]
                        for _ in range(cfg["num_epochs"] + 1)]
    expdata = {}
    for epoch in range(cfg["num_epochs"] + 1):  # +1 so that we have 250th epoch
        for exp in train_experiments:
            key, subkey = jax.random.split(key)
            (_loss, aux), (theta_grads, weights_grads) = loss_value_and_grad(
                subkey,  # Pass subkey this time, because loss will not return key
                exp.input_weights,
                exp.init_fixed_weights, # per-experiment arrays of fixed layers
                exp.feedforward_mask_training,
                exp.recurrent_mask_training,
                params['theta'],  # Current plasticity coeffs, updated on each iteration
                params['weights'],  # Current initial weights, updated on each iteration
                plasticity_func,  # Static within losses
                exp.data,
                exp.step_mask,
                exp.exp_i,
                cfg,  # Static within losses
                mode=('training' if not cfg._return_weights_trajec
                      else 'evaluation')  # Return trajectories in aux for debugging
            )
            # For debugging: return activation trajectory of each experiment
            _activation_trajs[epoch][exp.exp_i] = aux['trajectories']

            grads = {'theta': theta_grads, 'weights': weights_grads}

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        if epoch % cfg.log_interval == 0:
            print(f"Epoch {epoch}")
            expdata = evaluation.evaluate(
                key, cfg, params['theta'], plasticity_func,
                train_experiments, params['weights'],
                test_experiments, init_trainable_weights_test,
                expdata)

    return params, plasticity_func, expdata, _activation_trajs

def evaluate_model(
    key,
    cfg,
    test_experiments,
    learned_params,
    plasticity_func,
    expdata
):
    """Evaluate the trained model."""
    if cfg["num_exp_test"] == 0:
        return expdata

    r2_score = {"weights": [], "activity": []}
    percent_deviance = []

    for exp in test_experiments:
        key, simulation_key = jax.random.split(key)

        # Simulate model with learned_theta (plasticity coefficients)
        simulated_model_data = model.simulate_trajectory(
            simulation_key,
            exp.input_weights,
            exp.new_init_weights,
            exp.feedforward_mask_training,
            exp.recurrent_mask_training,
            learned_params['theta'],  # Learned plasticity coefficients estimate
            plasticity_func,
            exp.data,
            exp.step_mask,
            cfg,
            mode='generation_test'
        )

        # Simulate model with zeros plasticity coefficients for null model
        zero_theta, zero_plasticity_func = synapse.init_plasticity_volterra(
            key=None, init="zeros", scale=None
            )
        simulated_null_data = model.simulate_trajectory(
            simulation_key,
            exp.input_weights,
            exp.new_init_weights,  # Use the same initial weights as for learned model
            exp.feedforward_mask_training,
            exp.recurrent_mask_training,
            zero_theta,  # Zero plasticity coefficients
            zero_plasticity_func,
            exp.data,
            exp.step_mask,
            cfg,
            mode='generation_test'
        )

        percent_deviance.append(
            evaluate_percent_deviance(
                exp.data, simulated_model_data, simulated_null_data,
                exp.step_mask, mode=cfg.fit_data
            )
        )

        r2_score_exp = evaluate_r2_score(
                exp.step_mask,
                exp.data,
                exp.weights_trajec,
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
                              step_mask,
                              mode='neural'):
    """ Percent deviance explained by model.
    Calculate neg log likelihoods between model and data.
    Calculate neg log likelihoods between zero model and data.
    Percent deviance explained: (D_null - D_model) / D_null

    Args:
        data (dict): Dictionary of experimental data.
        simulated_model_data: Simulated activations with learned coefficients.
        simulated_null_data: Simulated activations with zero coefficients.
        step_mask: Mask of valid steps.
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
    step_mask = step_mask.flatten().astype(bool)

    # Choose only valid steps
    exp_activations = exp_activations[step_mask]
    model_activations = model_activations[step_mask]
    null_activations = null_activations[step_mask]

    if mode == 'behavioral':
        model_deviance = utils.binary_deviance(model_activations, exp_activations)
        null_deviance = utils.binary_deviance(null_activations, exp_activations)
    elif mode == 'neural':
        model_deviance = utils.sse_deviance(model_activations, exp_activations)
        null_deviance = utils.sse_deviance(null_activations, exp_activations)

    eps = 1e-12
    percent_deviance = 100 * (null_deviance - model_deviance) / (null_deviance + eps)
    return percent_deviance

def evaluate_r2_score(step_mask,
                      exp_data,
                      exp_weight_traj,
                      model_data,
                      cfg
):
    """
    Functionality: Evaluates the R2 score for weights and activity.
    Args:
        step_mask (N_sessions, N_steps_per_session_max),
        exp_data: Dict of (N_sessions, N_steps_per_session_max, ...) tensors,
        exp_weight_traj {  # Only if not cfg.use_experimental_data and only plastic
            'w_ff': (N_sessions, N_steps_per_session_max, N_hidden_pre, N_hidden_post),
            'w_rec': (N_sessions, N_steps_per_session_max, N_hidden_post, N_hidden_post
            ),
        model_data: Tuple of (x, y, output, weights) from model.simulate_trajectory,
        cfg

    Returns:
        Dict of R2 scores for activity (and weights).
    """
    r2_score = {}
    step_mask = step_mask.flatten().astype(bool)

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
    exp_activations = exp_activations[step_mask]
    model_activations = model_activations[step_mask]

    # Convert to numpy for sklearn
    exp_activations = np.asarray(jax.device_get(exp_activations))
    model_activations = np.asarray(jax.device_get(model_activations))

    r2_score["activity"] = sklearn.metrics.r2_score(exp_activations,
                                                    model_activations,
                                                    multioutput='variance_weighted')

    if not cfg.use_experimental_data:
        exp_weight_trajec = jnp.vstack(  # All plastic weight trajectories
            list(exp_weight_traj.values()))
        model_weight_trajec = jnp.vstack(
            list(model_data['weights'].values()))  # TODO b?

        # (N_sessions, N_steps_per_session_max, N_hidden_pre, N_hidden_post) ->
        # (N_steps_per_experiment, N_hidden_pre * N_hidden_post)
        exp_weight_trajec = exp_weight_trajec.reshape(
            -1, np.prod(exp_weight_trajec.shape[2:]))
        model_weight_trajec = model_weight_trajec.reshape(
            -1, np.prod(model_weight_trajec.shape[2:]))

        # Choose valid steps
        step_mask = jnp.repeat(step_mask, len(cfg.plasticity_layers))
        exp_weight_trajec = exp_weight_trajec[step_mask]
        model_weight_trajec = model_weight_trajec[step_mask]

        # Convert to numpy for sklearn
        exp_weight_trajec = np.asarray(jax.device_get(exp_weight_trajec))
        model_weight_trajec = np.asarray(jax.device_get(model_weight_trajec))

        r2_score["weights"] = sklearn.metrics.r2_score(exp_weight_trajec,
                                                       model_weight_trajec,
                                                       multioutput='variance_weighted')
    return r2_score

def save_results(cfg, expdata, train_time):
    """Save training logs and parameters."""
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

    _logdata_path = utils.save_logs(cfg, df)
