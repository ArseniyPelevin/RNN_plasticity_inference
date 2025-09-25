import evaluation
import jax
import jax.numpy as jnp
import losses
import model
import omegaconf
import optax
import pandas as pd
import synapse
import utils


def initialize_trainable_weights(key, cfg, num_experiments, n_restarts=1):
    """ Initialize new initial weights of trainable layers for each experiment.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary.
        num_experiments (int): Number of experiments to initialize weights for.
        n_restarts (int): Number of random trainable weights initializations
            per experiment. Default is 1 for training, can be >1 for evaluation.

    Returns:
        init_trainable_weights (dict): per-layer dict of arrays
            of shape (n_restarts, num_experiments, ...)
    """
    # Presplit keys for each restart and experiment
    keys = jax.random.split(key, n_restarts * num_experiments)
    keys = keys.reshape((n_restarts, num_experiments, 2))

    init_trainable_weights = {layer: [[] for _ in range(n_restarts)] 
                              for layer in cfg.trainable_init_weights}
    
    # Different initial weights for each restart of evaluation on test experiments
    for start in range(n_restarts):
        # Different initial weights for each simulated experiment
        for exp in range(num_experiments):
            weights = model.initialize_weights(keys[start, exp], 
                                               cfg, 
                                               cfg.init_weights_std_training, 
                                               layers=cfg.trainable_init_weights)
            for layer, layer_val in weights.items():
                init_trainable_weights[layer][start].append(layer_val)

    # Convert lists to arrays
    init_trainable_weights = {k: jnp.array(v)
                              for k, v in init_trainable_weights.items()}

    return init_trainable_weights

def train(key, cfg, train_experiments, test_experiments):
    """ Initialize values and functions, start training loop.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary.
        train_experiments (list): List of dicts of training experiments.
        test_experiments (list): List of dicts of test experiments.
    """
    key, init_plasticity_key, train_key, test_key = jax.random.split(key, 4)

    # Initialize plasticity coefficients for training
    init_theta, plasticity_func = synapse.init_plasticity(
        init_plasticity_key, cfg, mode="plasticity_model"
    )

    # Initialize weights of trainable layers for each train and test experiment. 
    # Store them as per-layer dict of arrays of shape (n_restarts, n_experiments, ...)
    init_trainable_weights_train = initialize_trainable_weights(
        train_key, cfg, cfg.num_train_experiments)
    init_trainable_weights_test = initialize_trainable_weights(
        test_key, cfg, cfg.num_test_experiments, n_restarts=cfg.num_test_restarts
    )

    params = {'theta': init_theta,
              'weights': init_trainable_weights_train[0]}  # n_restarts=1 for training

    # Return value (scalar) of the function (loss value) and gradient wrt its
    # parameters at argnums (theta and init_weights) - !Check argnums!
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=(4, 5), has_aux=True)

    # optimizer = optax.adam(learning_rate=cfg["learning_rate"])
    # Apply gradient clipping as in the article
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(learning_rate=cfg["learning_rate"]),
    )

    opt_state = optimizer.init(params)

    # Save and return deviances and R2s for each test sample at each evaluation epoch
    _losses_and_r2s = {}

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
                exp['init_fixed_weights'], # per-experiment arrays of fixed layers
                exp['feedforward_mask_training'],
                exp['recurrent_mask_training'],
                params['theta'],  # Current plasticity coeffs, updated on each iteration
                params['weights'],  # Current initial weights, updated on each iteration
                plasticity_func,  # Static within losses
                exp['data'],
                exp['rewarded_pos'],
                exp['step_mask'],
                exp['exp_i'],
                cfg,  # Static within losses
                mode=('training' if not cfg._return_weights_trajec
                      else 'evaluation')  # Return trajectories in aux for debugging
            )
            # For debugging: return activation trajectory of each experiment
            _activation_trajs[epoch][exp['exp_i']] = aux['trajectories']

            grads = {'theta': theta_grads, 'weights': weights_grads}

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        if epoch % cfg.log_interval == 0:
            print(f"\nEpoch {epoch}")
            expdata.setdefault("epoch", []).append(epoch)
            expdata = utils.print_and_log_learned_params(cfg, expdata, params['theta'])

            key, eval_key = jax.random.split(key)
            expdata, losses_and_r2 = evaluation.evaluate(
                eval_key, cfg, params['theta'], plasticity_func, init_theta,
                train_experiments, params['weights'],
                test_experiments, init_trainable_weights_test,
                expdata)
            _losses_and_r2s[epoch] = losses_and_r2

    return expdata, _activation_trajs, _losses_and_r2s

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
