from functools import partial

import evaluation
import jax
import jax.numpy as jnp
import losses
import model
import optax
import synapse
import utils


def train(key, cfg, train_experiments, test_experiments):
    """ Initialize values and functions, start training loop.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg (dict): Configuration dictionary.
        train_experiments (dict): Dictionary of arrays
            of shape (N_exp, ... ) for each variable of training experiments.
        test_experiments (dict): Dictionary of arrays
            of shape (N_exp, ... ) for each variable of test experiments.
    """
    num_epochs = cfg["num_epochs"] + 1  # +1 so that we have 250th epoch
    num_exp = cfg.num_exp_train
    (init_plasticity_key,
     train_w_key,
     test_w_key,
     *train_keys) = jax.random.split(key, 3 + num_epochs * num_exp)
    train_keys = jnp.array(train_keys).reshape((num_epochs, num_exp, 2))

    # Initialize plasticity coefficients for training
    init_theta, plasticity_func = synapse.init_plasticity(
        init_plasticity_key, cfg, mode="plasticity_model"
    )

    # Initialize weights of trainable layers for each train and test experiment.
    # Store them as per-layer dict of arrays of shape (n_restarts, n_experiments, ...)
    init_trainable_weights_train = model.initialize_trainable_weights(
        train_w_key, cfg, cfg.num_exp_train)
    init_trainable_weights_test = model.initialize_trainable_weights(
        test_w_key, cfg, cfg.num_exp_test, n_restarts=cfg.num_test_restarts)

    params = {'theta': init_theta,
              'weights': init_trainable_weights_train[0]}  # n_restarts=1 for training

    # Return value (scalar) of the function (loss value) and gradient wrt its
    # parameters at argnums (theta and init_weights) - !Check argnums!
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=(1, 2), has_aux=True)

    # optimizer = optax.adam(learning_rate=cfg["learning_rate"])
    # Apply gradient clipping as in the article
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["max_grad_norm"]),
        optax.adam(learning_rate=cfg["learning_rate"]),
    )
    opt_state = optimizer.init(params)

    expdata = {}  # Per-metric dict of per-evaluation-epoch lists
    _losses_and_r2s = {}  # Per-evaluation-epoch losses and r2 values (debugging only)
    _activation_trajs = []  # Per-epoch trajectories of activations (debugging only)

    @partial(jax.jit, static_argnames=('cfg'))
    def run_epoch(epoch_keys, cfg, params, opt_state, train_experiments):
        def run_exps(carry, exp_and_key):
            params, opt_state = carry
            exp, key = exp_and_key
            (loss, aux), (theta_grads, weights_grads) = loss_value_and_grad(
                key,  # Pass subkey this time, because loss will not return key
                params['theta'],  # Current plasticity coeffs, updated on each iteration
                params['weights'],  # Current initial weights, updated on each iteration
                plasticity_func,  # Static within losses
                exp,
                cfg,  # Static within losses
                mode=('training' if not cfg._return_weights_trajec
                      else 'evaluation')  # Return trajectories in aux for debugging
            )

            grads = {'theta': theta_grads, 'weights': weights_grads}
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # For debugging: return trajectories of activations and weights
            _activation_trajs = aux['trajectories']
            all_losses = {
                'total': loss,
                'neural': aux['neural'],
                'behavioral': aux['behavioral']
            }

            return (params, opt_state), (all_losses, _activation_trajs)

        (params, opt_state), (exps_losses, _activation_trajs) = jax.lax.scan(
            run_exps, (params, opt_state), (train_experiments, epoch_keys))

        # Return of jitted run_epoch()
        return params, opt_state, exps_losses, _activation_trajs

    for epoch in range(num_epochs):
        epoch_keys = train_keys[epoch]
        params, opt_state, exps_losses, _activation_trajs_epoch = run_epoch(
            epoch_keys, cfg, params, opt_state, train_experiments)

        _activation_trajs.append(_activation_trajs_epoch)  # Store for debugging

        if epoch % cfg.log_interval == 0:
            print(f"\nEpoch {epoch}")
            expdata.setdefault("epoch", []).append(epoch)

            # Print and log learned plasticity parameters
            expdata = utils.print_and_log_learned_params(cfg, expdata, params['theta'])

            # Print and log train loss
            train_loss_median = jnp.median(exps_losses['total'])
            print(f"Train Loss: {train_loss_median:.5f}")
            expdata.setdefault("train_loss_median", []).append(train_loss_median)

            key, eval_key = jax.random.split(key)
            expdata, _losses_and_r2 = evaluation.evaluate(
                eval_key, cfg, params['theta'], plasticity_func, init_theta,
                test_experiments, init_trainable_weights_test,
                expdata)
            _losses_and_r2s[epoch] = _losses_and_r2

    # Return of train()
    return expdata, _activation_trajs, _losses_and_r2s

