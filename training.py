import equinox as eqx
import evaluation
import jax
import jax.numpy as jnp
import losses
import optax
import plasticity
import utils

loss_value_and_grad = eqx.filter_value_and_grad(losses.loss, has_aux=True)

@eqx.filter_jit
def meta_learning_step(params, key, exp, plasticity, cfg, optimizer, opt_state):
    (loss, aux), grads = loss_value_and_grad(
        params,  # Current params on each iteration
        key,
        exp,
        plasticity,
        cfg,  # Static within losses
        returns=(None if not cfg.logging.log_trajectories
                 else ('xs', 'ys', 'outputs',
                       'decisions', 'rewards', 'weights'))
    )

    updates, opt_state = optimizer.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)

    return params, opt_state, loss, aux

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
    num_epochs = cfg.training.num_epochs + 1  # +1 so that we have 250th epoch
    num_exp_train = len(train_experiments)
    num_exp_test = len(test_experiments)

    # Presplit keys for initialization
    (init_plasticity_key,
     *train_keys) = jax.random.split(key, 1 + num_epochs * num_exp_train)
    train_keys = jnp.array(train_keys).reshape((num_epochs, num_exp_train) + key.shape)

    # Initialize plasticity modules for each plastic layer
    plasticity_train = plasticity.initialize_plasticity(
        init_plasticity_key, cfg.plasticity, mode='training')

    # Initialize trainable parameters
    # Initialize plasticity
    params = {'thetas': {layer: pl.coeffs for layer, pl in plasticity_train.items()},
              'w_init_learned': []}
    # Copy exp.w_init_train of trainable initial weights layers as starting points
    for exp in train_experiments:
        params['w_init_learned'].append(
            {layer: exp.w_init_train[layer]
             for layer in cfg.training.trainable_init_weights})

    # optimizer = optax.adam(learning_rate=cfg.learning_rate)
    # Apply gradient clipping as in the article
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.max_grad_norm),
        optax.adam(learning_rate=cfg.training.learning_rate),
    )
    opt_state = optimizer.init(params)

    expdata = {}  # Per-metric dict of per-evaluation-epoch lists
    _losses_and_r2s = {}  # Per-evaluation-epoch losses and r2 values (debugging only)
    _activation_trajs = {}  # Per-evaluation-epoch trajectories (debugging only)

    for epoch in range(num_epochs):
        epoch_keys = train_keys[epoch]
        epoch_trajectories, epoch_losses = [], []
        for exp_i, exp in enumerate(train_experiments):
            exp_key = epoch_keys[exp_i]

            params, opt_state, loss, aux = meta_learning_step(
                params, exp_key, exp, plasticity_train, cfg, optimizer, opt_state)

            # For debugging: return trajectories of activations and weights
            exp_trajectories = aux['trajectories']
            exp_losses = {
                'total': loss,
                'neural': aux['neural'],
                'behavioral': aux['behavioral']
            }
            if 'reinforcement' in cfg.training.fit_data:
                exp_losses['total_reward'] = aux['total_reward']
                exp_losses['total_licks'] = aux['total_licks']

            epoch_trajectories.append(exp_trajectories)
            epoch_losses.append(exp_losses)


        if epoch % cfg.logging.log_interval == 0:
            print(f"\nEpoch {epoch}")
            expdata.setdefault("epoch", []).append(epoch)

            _activation_trajs[epoch] = _activation_trajs_epoch  # Store for debugging

            # Print and log learned plasticity parameters
            expdata = utils.print_and_log_learned_params(cfg, expdata, params['thetas'])

            # Print and log train loss
            train_loss_median = jnp.median(exps_losses['total'])
            print(f"Train Loss: {train_loss_median:.5f}")
            expdata.setdefault("train_loss_median", []).append(train_loss_median)

            if 'reinforcement' in cfg.training.fit_data:
                train_rewards = jnp.median(exps_losses['total_reward'])
                train_licks = jnp.median(exps_losses['total_licks'])
                print(f"Train Rewards: {train_rewards:.1f}")
                print(f"Train Licks: {train_licks:.1f}")
                expdata.setdefault("train_rewards", []).append(train_rewards)
                expdata.setdefault("train_licks", []).append(train_licks)

            if not cfg.logging.do_evaluation:
                continue  # Skip evaluation on test set
            key, eval_key = jax.random.split(key)
            expdata, _losses_and_r2 = evaluation.evaluate(
                eval_key, cfg, params['thetas'], plasticity,
                test_experiments, w_init_test,
                expdata)
            _losses_and_r2s[epoch] = _losses_and_r2

    # Return of train()
    return params, expdata#, _activation_trajs, _losses_and_r2s  #, exps_losses
