from functools import partial

import jax
import jax.numpy as jnp
import losses
import numpy as np
import optax
import sklearn.metrics


def evaluate(key, cfg, theta, plasticity_func, init_theta,
             train_experiments, init_trainable_weights_train,
             test_experiments, init_trainable_weights_test,
             expdata):

    train_loss_key, losses_r2_key = jax.random.split(key, 2)

    # Evaluate train loss
    losses_and_r2_train = evaluate_loss(train_loss_key,
                                        cfg,
                                        train_experiments,
                                        plasticity_func,
                                        theta,
                                        init_trainable_weights_train,
                                        loss_only=True
                                        )
    train_loss_median = jnp.median(losses_and_r2_train['loss'])

    # Compute neural MSE loss and behavioral BCE loss.
    # Compute R2 scores for neural activity and weights.
    losses_and_r2 = compute_losses_and_r2(losses_r2_key, cfg,
                                          test_experiments, plasticity_func, init_theta,
                                          theta, init_trainable_weights_test)
    losses_and_r2_N = losses_and_r2.pop('N')  # Null model for reference

    # Extract test loss from dictionary
    test_loss_median = jnp.median(losses_and_r2['F']['loss'])

    # Evaluate percent deviance explained
    eps = 1e-12
    metric_for = {"neural": "MSE", "behavioral": "BCE"}
    PDE = {f'PDE_{model}_{data}': jnp.median(
        1 - (losses_and_r2[model][metric_for[data]] /
             (losses_and_r2_N[metric_for[data]] + eps))
        ) * 100
        for model in losses_and_r2
        for data in cfg.fit_data
        }

    # Evaluate R2 scores
    trajs = ['y', 'w'] if not cfg.use_experimental_data else ['y']
    R2 = {f'R2_{model}_{traj}': jnp.median(
        losses_and_r2[model][f'r2_{traj}'])
        for model in losses_and_r2
        for traj in trajs
    }

    # Print and log results
    print(f"Train Loss: {train_loss_median:.5f}")
    print(f"Test Loss: {test_loss_median:.5f}")
    expdata.setdefault("train_loss_median", []).append(train_loss_median)
    expdata.setdefault("test_loss_median", []).append(test_loss_median)
    # Log PDE
    for key, value in PDE.items():
        print(f"{key}: {value:.5f}")
        expdata.setdefault(key, []).append(value)
    # Log R2
    for model in losses_and_r2:
        r2_print_str = f"R2 {model}:\t"
        for traj in trajs:
            key = f'R2_{model}_{traj}'
            r2_print_str += f"{traj}: {R2[key]:.5f}\t"
            expdata.setdefault(key, []).append(R2[key])
        print(r2_print_str)


    return expdata, losses_and_r2

def compute_losses_and_r2(key, cfg, test_experiments, plasticity_func, init_theta,
                          theta, init_trainable_weights_test):
    """ Compute losses and R2 scores for different model variants:
    Full model (F): learned plasticity and learned weights,
    Theta model (T): learned plasticity and random weights,
    Weights model (W): zero plasticity and learned weights,
    Null model (N): zero plasticity and random weights.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        cfg (dict): Configuration dictionary.
        test_experiments (list): List of test experiments from class Experiment.
        plasticity_func (function): Plasticity function.
        theta (jax.numpy.ndarray): Learned plasticity coefficients.
        init_trainable_weights_test: Random initial trainable weights
            for test experiments.  Per-restart list of per-layer dicts
            of per-exp arrays of randomly initialized weights.

        Returns: dict: Dictionary with losses and R2 scores for each model variant.
    """
    # zero_theta, _ = synapse.init_plasticity_volterra(key=None,
    #                                                  init="zeros", scale=None)

    losses_and_r2 = {}

    # Evaluate test loss for configured number of restarts.
    # Use different initial weights for each restart.
    # Use the same set of initial weights in each evaluation epoch.
    for start in range(cfg.num_test_restarts):
        key, weights_key, loss_key = jax.random.split(key, 3)
        # Learn initial weights for test experiments
        if len(cfg.trainable_init_weights) > 0:
            learned_init_weights = learn_initial_weights(
                weights_key, cfg, theta, plasticity_func,
                test_experiments, init_trainable_weights_test[start])
        else:
            learned_init_weights = init_trainable_weights_test[start]

        compute_loss_r2 = partial(evaluate_loss,
                                  loss_key, cfg, test_experiments, plasticity_func)

        # Compute loss of full model with learned plasticity and learned weights
        losses_and_r2.setdefault("F", []).append(
            compute_loss_r2(theta, learned_init_weights))

        if cfg.trainable_init_weights:
            # Compute loss of theta model with learned plasticity and random weights
            losses_and_r2.setdefault("T", []).append(
                compute_loss_r2(theta, init_trainable_weights_test[start]))

            # Compute loss of weights model with zero plasticity and learned weights
            losses_and_r2.setdefault("W", []).append(
                compute_loss_r2(init_theta, learned_init_weights))

        # Compute loss of null model with zero plasticity and random weights
        losses_and_r2.setdefault("N", []).append(
            compute_loss_r2(init_theta, init_trainable_weights_test[start]))

    # Convert lists of dicts to dict of arrays
    for model in losses_and_r2:
        losses_and_r2[model] = {
            metric: jnp.array(
                [losses_and_r2[model][i][metric]
                 for i in range(cfg.num_test_restarts)])
                 for metric in losses_and_r2[model][0]
            }

    return losses_and_r2

def learn_initial_weights(key, cfg, learned_theta, plasticity_func,
                          test_experiments,
                          init_weights):

    # Compute gradients of loss wrt initial weights only
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=5, has_aux=True)

    # Apply gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm_weights"]),
        optax.adam(learning_rate=cfg["learning_rate_weights"]),
    )

    opt_state = optimizer.init(init_weights)

    for _epoch in range(cfg["num_epochs_weights"]):
        for exp in test_experiments:
            key, subkey = jax.random.split(key)
            (_loss, _aux), w_grads = loss_value_and_grad(
                key,
                learned_theta,
                init_weights,
                plasticity_func,
                exp,
                cfg,
                mode=('training' if not cfg._return_weights_trajec else 'evaluation')
            )

            updates, opt_state = optimizer.update(w_grads, opt_state, init_weights)
            init_weights = optax.apply_updates(init_weights, updates)

    return init_weights

def evaluate_loss(key, cfg, experiments, plasticity_func,
                  theta, init_trainable_weights, loss_only=False):

    (losses_total,
     losses_neural,
     losses_behavioral,
     r2s_neural,
     r2s_weights) = (np.zeros(len(experiments)) for _ in range(5))

    for exp_i, exp in enumerate(experiments):
        key, subkey = jax.random.split(key)
        loss, aux = losses.loss(
            subkey,  # Pass subkey this time, because loss will not return key
            exp.init_fixed_weights, # per-experiment arrays of fixed layers
            exp.feedforward_mask_training,
            exp.recurrent_mask_training,

            theta,
            init_trainable_weights,
            plasticity_func,  # Static within losses
            exp,
            cfg,  # Static within losses
            mode='evaluation'
        )

        step_mask = exp['step_mask'].ravel().astype(bool)

        # Compute R2 on neural activity
        r2_neural = jnp.array(jnp.nan)
        if 'neural' in cfg.fit_data:  # TODO? Don't use for training, but evaluate
            exp_activity = exp['data']['ys']
            model_activity = aux['trajectories']['ys']
            exp_activity = exp_activity.reshape(
                int(np.prod(exp_activity.shape[:2])), -1)
            model_activity = model_activity.reshape(
                int(np.prod(model_activity.shape[:2])), -1)
            r2_neural = evaluate_r2_score(
                step_mask,
                exp_activity,
                model_activity)

        # Compute R2 on weights only if we have weight trajectories (simulated data)
        r2_weights = jnp.array(jnp.nan)
        if not cfg.use_experimental_data:
            exp_weights = exp['weights_trajec']
            model_weights = aux['trajectories']['weights']
            exp_weights = jnp.concatenate([layer.reshape(
                int(np.prod(layer.shape[:2])), -1)
                for layer in exp_weights.values()], axis=1)
            model_weights = jnp.concatenate([layer.reshape(
                int(np.prod(layer.shape[:2])), -1)
                for layer in model_weights.values()], axis=1)
            r2_weights = evaluate_r2_score(
                step_mask,
                exp_weights,
                model_weights
            )
            r2s_weights[exp_i] = r2_weights

    return {'loss': losses_total,
            'MSE': losses_neural,
            'BCE': losses_behavioral,
            'r2_y': r2s_neural,
            'r2_w': r2s_weights
            }

def evaluate_r2_score(step_mask,
                      exp_values,
                      model_values
                      ):
    """
    Evaluates the R2 score of trajectories of neural activity or weights.
    Args:
        step_mask (N_sessions, N_steps_per_session_max), dtype=float,
        exp_values (N_sessions*N_steps_per_session_max, ...),
        model_values (N_sessions*N_steps_per_session_max, ...)

    Returns:
        R2 score (float)
    """
    # (N_sessions, N_steps_per_session_max, ...) -> (N_steps_per_experiment, ...)
    mask = jnp.ravel(step_mask)[:, None]

    mean_exp = jnp.sum(exp_values * mask, axis=0) / jnp.sum(mask)

    ss_res = jnp.sum(((exp_values - model_values) ** 2) * mask)
    ss_tot = jnp.sum(((exp_values - mean_exp) ** 2) * mask)

    return jnp.where(ss_tot > 0.0,
                     1.0 - (ss_res / ss_tot),
                     0.0)
