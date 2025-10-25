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
def meta_learning_step(params, key, exp, plasticity,
                       cfg, optimizer, opt_state, returns=()):
    """ Compute gradients of one trajectory and update params. """
    (_loss, aux), grads = loss_value_and_grad(
        params,  # Current params on each iteration
        key,
        exp,
        plasticity,
        cfg,  # Static within losses
        returns
    )

    updates, opt_state = optimizer.update(grads, opt_state)
    params = eqx.apply_updates(params, updates)

    return params, opt_state, aux

def meta_learn_plasticity(key, cfg, train_experiments, test_experiments,
                          params=None, last_epoch=0):
    """ Initialize plasticity, params, optimizer, and start training loop.
    Learn plasticity coefficients and initial weights of trainable layers.

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        cfg: Configuration object.
        train_experiments (list): List of Experiment objects for training.
        test_experiments (list): List of Experiment objects for evaluation.
        params (dict): Initial parameters for training. Provided in continued learning:
            'thetas': dict of plasticity coefficients for each plastic layer,
            'w_init_learned': per-training-experiment list of
                dicts of initial weights for each trainable layer.
        last_epoch (int): Last (evaluation) epoch number from previous training,
            for continued learning.

    Returns:
        params (dict): Learned parameters after training:
            'thetas': dict of plasticity coefficients for each plastic layer,
            'w_init_learned': per-training-experiment list of
                dicts of initial weights for each trainable layer.
        expdata (dict): Per-metric dict of per-evaluation-epoch lists.
        trajectories (dict): Per-evaluation-epoch dict of per-experiment lists
            of per-variable dicts of trajectories (if logged).
        _losses_and_r2s (dict): Per-evaluation-epoch dict of per-model-variant dicts
            with losses and R2 scores for each test experiment. For debugging only.
        """

    init_plasticity_key, train_key = jax.random.split(key)

    # Initialize training plasticity
    plasticity_train = plasticity.initialize_plasticity(
        init_plasticity_key, cfg.plasticity,
        mode = 'training' if not cfg.training.same_init_thetas else 'generation')
    # Add coefficients to params
    if params is None:  # Provided in continued learning
        params = {'thetas': {layer: pl.coeffs
                             for layer, pl in plasticity_train.items()}}

    num_epochs = cfg.training.num_epochs

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.max_grad_norm),
        optax.adam(learning_rate=cfg.training.learning_rate),
    )
    returns = (() if not cfg.logging.log_trajectories
               else ('xs', 'ys', 'outputs',
                     'decisions', 'rewards', 'weights'))

    params, expdata, trajectories, _losses_and_r2s = training_loop(
        train_key, params, plasticity_train, train_experiments,
        num_epochs, optimizer, cfg,
        test_experiments, returns, last_epoch, do_log=True)

    return params, expdata, trajectories, _losses_and_r2s

def training_loop(key, params, plasticity, experiments,
                  num_epochs, optimizer, cfg,
                  test_experiments=None, returns=(), last_epoch=0, do_log=False):
    """ Training loop for meta-learning.
    Used for both plasticity coefficients learning and initial weights learning
    (for test experiments in evaluation).

    Args:
        key (jax.random.PRNGKey): Random key for initialization.
        params (dict): Initial parameters for training:
            {thetas} in plasticity coefficients learning,
            {}} in initial weights learning.
            Initial weights of trainable layers are copied from exp.w_init_train.
        plasticity (dict): Plasticity modules for each plastic layer.
        experiments (list): List of Experiment objects for training.
            train_experiments for plasticity learning,
            test_experiments for initial weights learning in evaluation.
        num_epochs (int),
        optimizer (optax.GradientTransformation), parameters set by caller,
        cfg: Configuration object.
        test_experiments (list): List of Experiment objects for evaluation.
        returns (tuple): Tuple of strings indicating which outputs to return.
        last_epoch (int): Last (evaluation) epoch number from previous training,
            for continued learning.
        do_log (bool): Whether to log metrics and trajectories.

    Returns:
        params (dict): Learned parameters after training.
        expdata (dict): Per-metric dict of per-evaluation-epoch lists.
        trajectories (dict): Per-evaluation-epoch dict of per-experiment lists
            of per-variable dicts of trajectories (if logged).
        _losses_and_r2s (dict): Per-evaluation-epoch dict of per-model-variant dicts
            with losses and R2 scores for each test experiment. For debugging only.
    """

    num_exps = len(experiments)

    # Presplit keys for initialization
    train_keys = jax.random.split(key, num_epochs * num_exps
                                  ).reshape((num_epochs, num_exps) + key.shape)

    if 'w_init_learned' not in params:  # Provided in continued learning
        # Initialize trainable weights by copying layers of exp.w_init_train
        params['w_init_learned'] = []
        for exp in experiments:
            params['w_init_learned'].append(
                {layer: exp.w_init_train[layer]
                for layer in cfg.training.trainable_init_weights})

    opt_state = optimizer.init(params)

    expdata = {}  # Per-metric dict of per-evaluation-epoch lists
    _losses_and_r2s = {}  # Per-evaluation-epoch losses and r2 values (debugging only)
    trajectories = {}  # Per-evaluation-epoch trajectories (debugging only)

    start_epoch = last_epoch + 1
    end_epoch = start_epoch + num_epochs

    for epoch in range(start_epoch, end_epoch):
        epoch_keys = train_keys[epoch - start_epoch]
        epoch_trajectories, epoch_losses = [], []
        for exp_i, exp in enumerate(experiments):
            exp_key = epoch_keys[exp_i]

            params, opt_state, aux = meta_learning_step(
                params, exp_key, exp, plasticity, cfg, optimizer, opt_state,
                returns)

            # For initial weights learning of test experiments in evaluation
            if not do_log:
                continue

            if cfg.logging.log_trajectories:
                epoch_trajectory = {**aux['trajectories'],
                                    'init_weights': params['w_init_learned'][exp_i]}
                epoch_trajectories.append(epoch_trajectory)

            epoch_losses.append(aux)

        # For initial weights learning of test experiments in evaluation
        if not do_log:
            return params

        if (epoch % cfg.logging.log_interval == 0
            or epoch == start_epoch or epoch == end_epoch - 1):
            print(f"\nEpoch {epoch}")
            expdata, _losses_and_r2 = compute_and_log_evaluation_metrics(
                exp_key, epoch, cfg, expdata, params,
                test_experiments, epoch_losses)
            _losses_and_r2s[epoch] = _losses_and_r2

            if cfg.logging.log_trajectories:
                trajectories[epoch] = epoch_trajectories

    # Return of train()
    return params, expdata, trajectories, _losses_and_r2s

def compute_and_log_evaluation_metrics(key, epoch, cfg, expdata,
                                       params,
                                       test_experiments,
                                       epoch_losses):
    """ Log metrics on training set and evaluate on test set. """

    expdata.setdefault("epoch", []).append(epoch)

    # Print and log learned plasticity parameters
    expdata = utils.print_and_log_learned_params(cfg, expdata, params['thetas'])

    # Print and log train loss
    train_loss_median = jnp.median(
        jnp.array([exp_loss['total'] for exp_loss in epoch_losses]))
    print(f"Train Loss: {train_loss_median:.5f}")
    expdata.setdefault("train_loss_median", []).append(train_loss_median)

    if 'reinforcement' in cfg.training.fit_data:
        train_rewards = jnp.median(
            jnp.array([exp_loss['total_reward'] for exp_loss in epoch_losses]))
        train_licks = jnp.median(
            jnp.array([exp_loss['total_licks'] for exp_loss in epoch_losses]))
        print(f"Train Rewards: {train_rewards:.1f}")
        print(f"Train Licks: {train_licks:.1f}")
        expdata.setdefault("train_rewards", []).append(train_rewards)
        expdata.setdefault("train_licks", []).append(train_licks)

    if not cfg.logging.do_evaluation:
        return expdata, None  # Skip evaluation on test set

    key, eval_key = jax.random.split(key)
    expdata, _losses_and_r2 = evaluation.evaluate(
        eval_key, params['thetas'], test_experiments, expdata, cfg)

    return expdata, _losses_and_r2

