from functools import partial

import jax
import jax.numpy as jnp
import losses
import numpy as np
import optax


def evaluate(key, cfg, theta, plasticity_func, init_theta,
             test_experiments, init_trainable_weights_test, expdata):
    """ Compute evaluation metrics.

    Args:
        key (jax.random.PRNGKey): Random key for simulation.
        cfg (dict): Configuration dictionary.
        theta (jax.numpy.ndarray): Learned plasticity coefficients.
        plasticity_func (function): Plasticity function.
        init_theta (jax.numpy.ndarray): Initial random plasticity coefficients.
        test_experiments (dict): Dictionary of arrays
            of shape (N_exp, ... ) for each variable of test experiments.
        init_trainable_weights_test: Random initial trainable weights
            for test experiments.  Per-restart list of per-layer dicts
            of per-exp arrays of randomly initialized weights.

    Returns:
        expdata (dict): Updated expdata dictionary with evaluation metrics.
        losses_and_r2 (dict): Dictionary with losses and R2 scores
            for each experiment in each model variant.
    """

    # Compute neural MSE loss and behavioral BCE loss.
    # Compute R2 scores for neural activity and weights.
    losses_and_r2 = compute_models_losses_and_r2(key, cfg, test_experiments,
                                                 plasticity_func, init_theta, theta,
                                                 init_trainable_weights_test)
    losses_and_r2_N = losses_and_r2.pop('N')  # Null model for reference

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

    # Print and log test loss
    test_loss_median = jnp.median(losses_and_r2['F']['loss'])
    print(f"Test Loss: {test_loss_median:.5f}")
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

def compute_models_losses_and_r2(key, cfg, test_experiments,
                                 plasticity_func, init_theta, theta,
                                 init_trainable_weights_test):
    """ Compute losses and R2 scores for different model variants:
    Full model (F): learned plasticity and learned weights,
    Theta model (T): learned plasticity and random weights,
    Weights model (W): zero plasticity and learned weights,
    Null model (N): zero plasticity and random weights.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        cfg (dict): Configuration dictionary.
        test_experiments (dict): Dictionary of arrays
            of shape (N_exp, ... ) for each variable of test experiments.
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
        if len(cfg.trainable_init_weights) > 0:
            # Learn initial weights for test experiments
            learned_init_weights = learn_initial_weights(
                weights_key, cfg, theta, plasticity_func,
                test_experiments, init_trainable_weights_test[start])
        else:
            # Use random initial weights for test experiments
            learned_init_weights = init_trainable_weights_test[start]

        compute_loss_r2 = partial(compute_loss_and_r2,
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

@partial(jax.jit, static_argnames=['cfg', 'plasticity_func'])
def learn_initial_weights(key, cfg, learned_theta, plasticity_func,
                          test_experiments,
                          init_weights):
    """ Learn initial weights of trainable layers for test experiments given theta.
    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        cfg: Configuration dictionary.
        learned_theta (jax.numpy.ndarray): Learned plasticity coefficients.
        plasticity_func (function): Plasticity function.
        test_experiments (dict): Variables in arrays
            of shape (N_exp, N_sess, N_steps, ...) for one start.
        init_weights (dict): Random initial trainable weights
            for test experiments. Per-layer dicts of per-exp arrays
            of randomly initialized weights for one restart.

    Returns:
        init_weights (dict): Learned initial trainable weights
            for test experiments. Per-layer dicts of per-exp arrays
            of learned weights for one restart.
    """
    # Presplit keys for each epoch and experiment
    test_keys = jax.random.split(key, cfg.num_epochs_weights * cfg.num_exp_test)
    test_keys = test_keys.reshape(cfg.num_epochs_weights, cfg.num_exp_test, 2)

    # Compute gradients of loss wrt initial weights only
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=2, has_aux=True)

    # Apply gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm_weights"]),
        optax.adam(learning_rate=cfg["learning_rate_weights"]),
    )

    opt_state = optimizer.init(init_weights)

    @partial(jax.jit, static_argnames=['plasticity_func', 'cfg'])
    def run_epoch(epoch_keys, init_weights, opt_state, learned_theta, plasticity_func,
                  test_experiments, cfg):
        def run_exps(carry, exp_and_key):
            init_weights, opt_state = carry
            exp, key = exp_and_key
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
            return (init_weights, opt_state), None

        (init_weights, opt_state), _ = jax.lax.scan(
            run_exps, (init_weights, opt_state), (test_experiments, epoch_keys))

        return init_weights, opt_state

    for epoch in range(cfg.num_epochs_weights):
        init_weights, opt_state = run_epoch(test_keys[epoch], init_weights, opt_state,
                                            learned_theta, plasticity_func,
                                            test_experiments, cfg)

    return init_weights

@partial(jax.jit, static_argnames=['cfg', 'plasticity_func'])
def compute_loss_and_r2(key, cfg, experiments, plasticity_func, theta,
                        init_trainable_weights):
    """ Compute loss and R2 scores for one model variant on all test experiments.

    Args:
        key (jax.random.PRNGKey): Random key for simulation.
        cfg: Configuration dictionary.
        experiments (dict): Variables in arrays
            of shape (N_exp, N_sess, N_steps, ...) for one start.
        plasticity_func (function): Plasticity function.
        theta (jax.numpy.ndarray): Plasticity coefficients.
        init_trainable_weights (dict): Initial synaptic weights in trainable layers.

    Returns:
        dict: Dictionary of per-experiment arrays:
            total loss, neural loss, behavioral loss,
            R2 score for neural activity and R2 score for weights (if simulated data).
    """

    keys = jax.random.split(key, cfg.num_exp_test)

    # Compute loss and R2 score for each experiment in a scan
    def run_exps(_, exp_and_key):
        exp, key = exp_and_key

        # Compute test loss and R2 score
        loss, aux = losses.loss(
            key,
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

        return None, (loss, aux['neural'], aux['behavioral'], r2_neural, r2_weights)

    _, (losses_total, losses_neural, losses_behavioral,
        r2s_neural, r2s_weights) = jax.lax.scan(run_exps, None, (experiments, keys))

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
