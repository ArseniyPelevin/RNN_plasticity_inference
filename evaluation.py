
from functools import partial

import jax
import jax.numpy as jnp
import losses
import numpy as np
import optax
import sklearn.metrics
import synapse


def evaluate(key, cfg, theta, plasticity_func,
             train_experiments, init_trainable_weights_train,
             test_experiments, init_trainable_weights_test,
             expdata):

    # Evaluate train loss
    losses_and_r2_train = evaluate_loss(key,  # TODO key
                                        cfg,
                                        train_experiments,
                                        plasticity_func,
                                        theta,
                                        init_trainable_weights_train,
                                        loss_only=True
                                        )
    train_loss_mean = jnp.mean(losses_and_r2_train['loss'])
    train_loss_std = jnp.std(losses_and_r2_train['loss'])

    # Compute neural MSE loss and/or behavioral BCE loss.
    # Compute R2 scores for neural activity and/or behavioral output and/or weights.
    losses_and_r2 = compute_losses_and_r2(key, cfg, test_experiments, plasticity_func,
                                          theta, init_trainable_weights_test)  # TODO key

    # Extract test loss from dictionary
    test_loss_mean = losses_and_r2['F']['loss_mean']
    test_loss_std = losses_and_r2['F']['loss_std']

    # Evaluate percent deviance explained


    # Evaluate R2


    return expdata

def compute_losses_and_r2(key, cfg, test_experiments, plasticity_func,
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
    zero_theta, _ = synapse.init_plasticity_volterra(key=None, init="zeros", scale=None)

    losses_and_r2 = {'F': [], 'T': [], 'W': [], 'N': []}

    # Evaluate test loss for configured number of restarts.
    # Use different initial weights for each restart.
    # Use the same set of initial weights in each evaluation epoch.
    for start in range(cfg.num_test_restarts):
        print(f"Test restart {start+1}/{cfg.num_test_restarts}")
        # Learn initial weights for test experiments
        learned_init_weights = learn_initial_weights(
            key, cfg, theta, plasticity_func,  # TODO key
            test_experiments, init_trainable_weights_test[start])

        compute_loss_r2 = partial(evaluate_loss,
                                  key, cfg, test_experiments, plasticity_func)

        # Compute loss of full model with learned plasticity and learned weights
        losses_and_r2['F'].append(compute_loss_r2(theta,
                                                  learned_init_weights))

        # Compute loss of theta model with learned plasticity and random weights
        losses_and_r2['T'].append(compute_loss_r2(theta,
                                                  init_trainable_weights_test[start]))

        # Compute loss of weights model with zero plasticity and learned weights
        losses_and_r2['W'].append(compute_loss_r2(zero_theta,
                                                  learned_init_weights))

        # Compute loss of null model with zero plasticity and random weights
        losses_and_r2['N'].append(compute_loss_r2(zero_theta,
                                                  init_trainable_weights_test[start]))

    # Average over restarts
    for mod in losses_and_r2:
        losses_and_r2[mod] = {
            f'{metric}_mean': jnp.mean(jnp.array(
                [losses_and_r2[mod][i][metric]
                 for i in range(cfg.num_test_restarts)]))
                 for metric in losses_and_r2[mod][0]} | {
            f'{metric}_std': jnp.std(jnp.array(
                [losses_and_r2[mod][i][metric]
                 for i in range(cfg.num_test_restarts)]))
                 for metric in losses_and_r2[mod][0]} | {
            f'{metric}_all': jnp.array(
                [losses_and_r2[mod][i][metric]
                 for i in range(cfg.num_test_restarts)])
                 for metric in losses_and_r2[mod][0]
            }

    return losses_and_r2

def learn_initial_weights(key, cfg, learned_theta, plasticity_func,
                          test_experiments,
                          init_weights):

    # Compute gradients of loss wrt initial weights only
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=6, has_aux=True)

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
                subkey,  # Pass subkey this time, because loss will not return key
                exp.input_weights,
                exp.init_fixed_weights, # per-experiment arrays of fixed layers
                exp.feedforward_mask_training,
                exp.recurrent_mask_training,
                learned_theta,  # Learned plasticity coefficients by this eval epoch
                init_weights,  # Current initial weights, to be optimized
                plasticity_func,  # Static within losses
                exp.data,
                exp.step_mask,
                exp.exp_i,
                cfg,  # Static within losses
                mode=('training' if not cfg._return_weights_trajec
                      else 'evaluation')  # Return trajectories in aux for debugging
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
     r2s_behavioral,
     r2s_weights) = (np.zeros(len(experiments)) for _ in range(6))

    for exp_i, exp in enumerate(experiments):
        key, subkey = jax.random.split(key)
        loss, aux = losses.loss(
            subkey,  # Pass subkey this time, because loss will not return key
            exp.input_weights,
            exp.init_fixed_weights, # per-experiment arrays of fixed layers
            exp.feedforward_mask_training,
            exp.recurrent_mask_training,

            theta,
            init_trainable_weights,

            plasticity_func,  # Static within losses
            exp.data,
            exp.step_mask,
            exp.exp_i,  # Internal index of the experiment
            cfg,  # Static within losses
            mode='evaluation'
        )

        losses_total[exp_i] = loss
        if loss_only:
            continue

        losses_neural[exp_i] = aux['neural']
        losses_behavioral[exp_i] = aux['behavioral']

        step_mask = exp.step_mask.flatten().astype(bool)

        if 'neural' in cfg.fit_data:
            r2_neural = evaluate_r2_score_activations(
                step_mask,
                exp.data['ys'],
                aux['trajectories']['ys'])
            r2s_neural[exp_i] = r2_neural

        if 'behavioral' in cfg.fit_data:
            r2_behavioral = evaluate_r2_score_activations(
                step_mask,
                exp.data['decisions'],
                aux['trajectories']['outputs'])
            r2s_behavioral[exp_i] = r2_behavioral

        if not cfg.use_experimental_data:
            r2_weights = evaluate_r2_score_weights(
                step_mask,
                exp.weights_trajec,
                aux['trajectories']['weights'],
                cfg
            )
            r2s_weights[exp_i] = r2_weights

    return {'loss': losses_total,
            'MSE': losses_neural,
            'BCE': losses_behavioral,
            'r2_y': r2s_neural,
            'r2_out': r2s_behavioral,
            'r2_w': r2s_weights
            }

def evaluate_r2_score_activations(step_mask,
                                  exp_activations,
                                  model_activations
                                  ):
    """
    Functionality: Evaluates the R2 score for activity.
    Args:
        step_mask (N_sessions, N_steps_per_session_max),
        exp_activations (N_sessions, N_steps_per_session_max, N_neurons or 1),
        model_activations (N_sessions, N_steps_per_session_max, N_neurons or 1)

    Returns:
        R2 score for activity, variance-weighted
    """

    # (N_sessions, N_steps_per_session_max, ...) -> (N_steps_per_experiment, ...)
    exp_activations = exp_activations.reshape(-1, *exp_activations.shape[2:])
    model_activations = model_activations.reshape(-1, *model_activations.shape[2:])

    # Choose valid steps
    exp_activations = exp_activations[step_mask]
    model_activations = model_activations[step_mask]

    # Convert to numpy for sklearn
    exp_activations = np.asarray(jax.device_get(exp_activations))
    model_activations = np.asarray(jax.device_get(model_activations))

    return sklearn.metrics.r2_score(exp_activations,
                                    model_activations,
                                    multioutput='variance_weighted')

def evaluate_r2_score_weights(step_mask,
                              exp_weight_traj,
                              model_weight_traj,
                              cfg
                              ):
    """
    Functionality: Evaluates the R2 score for weights.
    Args:
        step_mask (N_sessions, N_steps_per_session_max),
        exp_weight_traj {  # Only if not cfg.use_experimental_data and only plastic
            'w_ff': (N_sessions, N_steps_per_session_max, N_hidden_pre, N_hidden_post),
            'w_rec': (N_sessions, N_steps_per_session_max, N_hidden_post, N_hidden_post
            ),
        model_weight_traj,
        cfg

    Returns:
        R2 scores for weights, variance-weighted
    """
    exp_weight_trajec = jnp.vstack(  # All plastic weight trajectories
        list(exp_weight_traj.values()))
    model_weight_trajec = jnp.vstack(
        list(model_weight_traj.values()))  # TODO b?

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

    return sklearn.metrics.r2_score(exp_weight_trajec,
                                    model_weight_trajec,
                                    multioutput='variance_weighted')
