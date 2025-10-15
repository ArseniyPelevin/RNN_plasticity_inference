from functools import partial

import evaluation
import jax
import jax.numpy as jnp
import losses
import network
import optax
import plasticity
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
    num_epochs = cfg.num_epochs + 1  # +1 so that we have 250th epoch
    num_exp_train = len(train_experiments)
    num_exp_test = len(test_experiments)

    # Presplit keys for initialization
    (init_plasticity_key,
     train_w_key,
     test_w_key,
     *train_keys) = jax.random.split(key, 3 + num_epochs * num_exp_train)
    train_keys = jnp.array(train_keys).reshape((num_epochs, num_exp_train) + key.shape)

    # Initialize plasticity modules for each plastic layer
    plasticity_train = plasticity.initialize_plasticity(
        init_plasticity_key, cfg.plasticity, mode='training')
    # Initialize weights of all layers for each train and test experiment.
    # Store them as per-layer dict of arrays of shape (n_experiments, ...)
    w_init_train = network.initialize_weights(
        train_w_key, len(train_experiments), cfg.network)
    w_init_test = network.initialize_weights(
        test_w_key, len(test_experiments), cfg.network)

    # Apply initial weights to each experiment's network
    for i, exp in enumerate(train_experiments):
        exp.network = exp.network.apply_weights(w_init_train[i])
    # TODO test

    # Initialize trainable parameters
    params = {'plasticity': plasticity_train,
              'w_init_learned': {layer: w_init_train[layer]
                                 for layer in cfg.training.trainable_init_weights}}

    # Return value (scalar) of the function (loss value) and gradient wrt its
    # parameters at argnums (theta and init_weights) - !Check argnums!
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=(1, 2), has_aux=True)

    # optimizer = optax.adam(learning_rate=cfg.learning_rate)
    # Apply gradient clipping as in the article
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=cfg.learning_rate),
    )
    opt_state = optimizer.init(params)

    expdata = {}  # Per-metric dict of per-evaluation-epoch lists
    _losses_and_r2s = {}  # Per-evaluation-epoch losses and r2 values (debugging only)
    _activation_trajs = {}  # Per-evaluation-epoch trajectories (debugging only)

    @partial(jax.jit, static_argnames=('cfg'))
    def run_epoch(epoch_keys, cfg, params, opt_state, train_experiments):
        def run_exps(carry, exp_and_key):
            params, opt_state = carry
            exp, key = exp_and_key
            (loss, aux), (theta_grads, weights_grads) = loss_value_and_grad(
                key,  # Pass subkey this time, because loss will not return key
                params['thetas'],  # Current plasticity coeffs on each iteration
                params['weights'],  # Current initial weights on each iteration
                exp,
                cfg,  # Static within losses
                returns=(None if not cfg.log_trajectories
                         else ('xs', 'ys', 'outputs',
                               'decisions', 'rewards', 'weights'))
            )

            grads = {'thetas': theta_grads, 'weights': weights_grads}
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # For debugging: return trajectories of activations and weights
            _activation_trajs = aux['trajectories']
            all_losses = {
                'total': loss,
                'neural': aux['neural'],
                'behavioral': aux['behavioral']
            }
            if 'reinforcement' in cfg.fit_data:
                all_losses['total_reward'] = aux['total_reward']
                all_losses['total_licks'] = aux['total_licks']

            return (params, opt_state), (all_losses, _activation_trajs)

        (params, opt_state), (exps_losses, _activation_trajs) = jax.lax.scan(
            run_exps, (params, opt_state), (train_experiments, epoch_keys))

        # Return of jitted run_epoch()
        return params, opt_state, exps_losses, _activation_trajs

    for epoch in range(num_epochs):
        epoch_keys = train_keys[epoch]
        params, opt_state, exps_losses, _activation_trajs_epoch = run_epoch(
            epoch_keys, cfg, params, opt_state, train_experiments)

        if epoch % cfg.log_interval == 0:
            print(f"\nEpoch {epoch}")
            expdata.setdefault("epoch", []).append(epoch)

            _activation_trajs[epoch] = _activation_trajs_epoch  # Store for debugging

            # Print and log learned plasticity parameters
            expdata = utils.print_and_log_learned_params(cfg, expdata, params['thetas'])

            # Print and log train loss
            train_loss_median = jnp.median(exps_losses['total'])
            print(f"Train Loss: {train_loss_median:.5f}")
            expdata.setdefault("train_loss_median", []).append(train_loss_median)

            if 'reinforcement' in cfg.fit_data:
                train_rewards = jnp.median(exps_losses['total_reward'])
                train_licks = jnp.median(exps_losses['total_licks'])
                print(f"Train Rewards: {train_rewards:.1f}")
                print(f"Train Licks: {train_licks:.1f}")
                expdata.setdefault("train_rewards", []).append(train_rewards)
                expdata.setdefault("train_licks", []).append(train_licks)

            if not cfg.do_evaluation:
                continue  # Skip evaluation on test set
            key, eval_key = jax.random.split(key)
            expdata, _losses_and_r2 = evaluation.evaluate(
                eval_key, cfg, params['thetas'], plasticity,
                test_experiments, w_init_test,
                expdata)
            _losses_and_r2s[epoch] = _losses_and_r2

    # Return of train()
    return params, expdata, _activation_trajs, _losses_and_r2s  #, exps_losses
