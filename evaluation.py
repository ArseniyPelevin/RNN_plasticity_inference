from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import losses
import numpy as np
import optax
import plasticity
import training


def evaluate(key, thetas, test_experiments, expdata, cfg):
    """ Compute evaluation metrics.

    Args:
        key (jax.random.PRNGKey): Random key for simulation.
        thetas (dict): Per-plastic-layer learned plasticity coefficients.
        test_experiments (list): List of Experiment objects for evaluation.
        cfg (dict): Configuration dictionary.

    Returns:
        expdata (dict): Updated expdata dictionary with evaluation metrics.
        losses_and_r2 (dict): Dictionary for different model variants
            with losses and R2 scores for each test experiment.
    """

    # Compute neural MSE loss and behavioral BCE loss.
    # Compute R2 scores for neural activity and weights.
    losses_and_r2 = compute_models_losses_and_r2(key, thetas,
                                                 test_experiments,
                                                 cfg)
    losses_and_r2_N = losses_and_r2.pop('N')  # Null model for reference

    # Log test loss
    test_loss_median = jnp.median(losses_and_r2['F']['loss'])
    print(f"Test Loss: {test_loss_median:.5f}")
    expdata.setdefault("test_loss_median", []).append(test_loss_median)

    # Evaluate and log reinforcement metrics: total rewards and total licks
    if "reinforcement" in cfg.training.fit_data:
        for model in losses_and_r2:
            rewards = jnp.median(losses_and_r2[model]['Rewards'])
            licks = jnp.median(losses_and_r2[model]['Licks'])
            print(f"{model} - Rewards: {rewards:.1f}, Licks: {licks:.1f}")
            expdata.setdefault(f'Rewards_{model}', []).append(rewards)
            expdata.setdefault(f'Licks_{model}', []).append(licks)

        # Other metrics are meaningless for reinforcement learning
        return expdata, losses_and_r2

    # Evaluate percent deviance explained
    eps = 1e-12
    metric_for = {"neural": "MSE", "behavioral": "BCE"}
    PDE = {f'PDE_{model}_{data}': jnp.median(
        1 - (losses_and_r2[model][metric_for[data]] /
             (losses_and_r2_N[metric_for[data]] + eps))
        ) * 100
        for model in losses_and_r2
        for data in cfg.training.fit_data
        }

    # Evaluate R2 scores
    trajs = ['y', 'w'] if not cfg.experiment.use_experimental_data else ['y']
    R2 = {f'R2_{model}_{traj}': jnp.median(
        losses_and_r2[model][f'r2_{traj}'])
        for model in losses_and_r2
        for traj in trajs
    }

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

def compute_models_losses_and_r2(key, thetas, test_experiments, cfg):
    """ Compute losses and R2 scores for different model variants:
    Full model (F): learned plasticity and learned weights,
    Theta model (T): learned plasticity and random weights,
    Weights model (W): zero plasticity and learned weights,
    Null model (N): zero plasticity and random weights.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        thetas (dict): Per-plastic-layer dict of learned plasticity coeffs.
        test_experiments (list): List of Experiment objects for evaluation.
        cfg (dict): Configuration dictionary.

    Returns: dict: Dictionary with losses and R2 scores for each model variant.
    """
    plasticity_key, weights_key, loss_key = jax.random.split(key, 3)

    plasticity_test = plasticity.initialize_plasticity(
        plasticity_key, cfg.plasticity, mode='evaluation', init_scale=0.0)

    null_thetas = {layer: plasticity_test[layer].coeffs for layer in plasticity_test}
    null_w_init = [{} for _ in range(len(test_experiments))]

    # # Evaluate test loss for configured number of restarts.
    # # Use different initial weights for each restart.
    # # Use the same set of initial weights in each evaluation epoch.
    # for start in range(cfg.num_test_restarts):

    if len(cfg.training.trainable_init_weights) > 0:
        # Learn initial weights for test experiments
        w_init_learned = meta_learn_test_initial_weights(
            weights_key, thetas, plasticity_test, test_experiments, cfg)
    else:
        w_init_learned = [{} for _ in range(len(test_experiments))]

    compute_loss_r2 = partial(compute_loss_and_r2,
                              loss_key, plasticity_test, test_experiments, cfg)

    losses_and_r2 = {}

    # Compute loss of full model with learned plasticity and learned weights
    losses_and_r2["F"] = compute_loss_r2(thetas, w_init_learned)

    if cfg.training.trainable_init_weights:
        # Compute loss of theta model with learned plasticity and random weights
        losses_and_r2["T"] = compute_loss_r2(thetas, null_w_init)

        # Compute loss of weights model with zero plasticity and learned weights
        losses_and_r2["W"] = compute_loss_r2(null_thetas, w_init_learned)

    # Compute loss of null model with zero plasticity and random weights
    losses_and_r2["N"] = compute_loss_r2(null_thetas, null_w_init)

    return losses_and_r2

def meta_learn_test_initial_weights(key, learned_thetas, plasticity,
                                    test_experiments, cfg):
    """ Learn initial weights of trainable layers for test experiments given theta.
    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        learned_thetas (dict): Per-plastic-layer dict of learned plasticity coeffs.
        plasticity (dict): Per-plastic-layer dict of plasticity modules.
        test_experiments (list): List of Experiment objects for evaluation.
        cfg (dict): Configuration dictionary.

    Returns:
        init_weights (dict): Learned initial trainable weights
            for test experiments. Per-layer dicts of per-exp arrays
            of learned weights for one restart.
    """

    # Set fixed plasticity coefficients from learned thetas
    for layer in learned_thetas:
        # Apply mask to plasticity coefficients to enforce constraints
        if cfg.plasticity.plasticity_models[layer] == "volterra":
            learned_thetas[layer] *= plasticity[layer].coeff_mask
        # Apply current coefficients estimates to plasticity modules
        plasticity[layer] = eqx.tree_at(
            lambda p: p.coeffs, plasticity[layer], learned_thetas[layer])

    params = {}  # No theta in params for initial weights learning
    # Weights will be taken from exp.w_init_train inside training loop

    num_epochs = cfg.training.num_epochs_weights

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.max_grad_norm_weights),
        optax.adam(learning_rate=cfg.training.learning_rate_weights),
    )

    params = training.training_loop(key, params, plasticity, test_experiments,
                                    num_epochs, optimizer, cfg)
    return params['w_init_learned']

@eqx.filter_jit
def compute_loss_and_r2(key, plasticity, experiments, cfg,
                        thetas, w_init):
    """ Compute loss and R2 scores for one model variant on all test experiments.

    Args:
            # Same for all model variants:
        key (jax.random.PRNGKey): Random key for simulation.
        plasticity (dict): Per-plastic-layer dict of plasticity modules.
        experiments (list): List of Experiment objects for evaluation.
        cfg (dict): Configuration dictionary.
            # Different for each model variant:
        thetas (dict): Per-plastic-layer dict of plasticity coefficients.
        init_trainable_weights (dict): Initial synaptic weights in trainable layers.

    Returns:
        dict: Per-metric dictionary of per-experiment arrays of losses and R2 scores.:
            total loss, neural loss, behavioral loss,
            R2 score for neural activity and R2 score for weights (if simulated data).
    """
    keys = jax.random.split(key, len(experiments))

    params = {'thetas': thetas, 'w_init_learned': w_init}

    metrics = {metric: [] for metric in ['loss', 'MSE', 'BCE',
                                         'Rewards', 'Licks', 'r2_y', 'r2_w']}
    # Compute loss and R2 score for each experiment
    for exp_i, exp in enumerate(experiments):
        # Compute test loss and R2 score
        loss, aux = losses.loss(
            params,  # Current params on each iteration
            keys[exp_i],
            exp,
            plasticity,
            cfg,  # Static within losses
            returns=('ys' if 'neural' in cfg.training.fit_data else '',
                     'weights' if not cfg.experiment.use_experimental_data else '')
        )

        step_mask = exp.step_mask.ravel().astype(bool)

        # Compute R2 on neural activity
        r2_neural = jnp.array(jnp.nan)  # TODO? Don't use for training, but evaluate
        if 'neural' in cfg.training.fit_data:
            exp_activity = exp.data['ys']
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
        if (not cfg.experiment.use_experimental_data and
            'reinforcement' not in cfg.training.fit_data):
            exp_weights = exp.weights_trajec
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

        metrics['loss'].append(loss)
        metrics['MSE'].append(aux['neural'])
        metrics['BCE'].append(aux['behavioral'])
        metrics['Rewards'].append(aux['total_reward'])
        metrics['Licks'].append(aux['total_licks'])
        metrics['r2_y'].append(r2_neural)
        metrics['r2_w'].append(r2_weights)

    return {k: jnp.array(v) for k, v in metrics.items()}

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
