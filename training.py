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

def meta_learn_plasticity(key, cfg, train_experiments, test_experiments):
    """ Initialize plasticity, params, optimizer, and start training loop.
    Learn plasticity coefficients and initial weights of trainable layers. """

    init_plasticity_key, train_key = jax.random.split(key)

    # Initialize plasticity and add coefficients to params
    plasticity_train = plasticity.initialize_plasticity(
        init_plasticity_key, cfg.plasticity, mode='training')
    params = {'thetas': {layer: pl.coeffs for layer, pl in plasticity_train.items()}}

    num_epochs = cfg.training.num_epochs + 1  # +1 so that we have 250th epoch

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
        test_experiments, returns, do_log=True)

    return params, expdata, trajectories, _losses_and_r2s

def training_loop(key, params, plasticity, experiments,
                  num_epochs, optimizer, cfg,
                  test_experiments=None, returns=(), do_log=False):
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
        do_log (bool): Whether to log metrics and trajectories.

    Returns:
        params (dict): Learned parameters after training.
        expdata (dict): Per-metric dict of per-evaluation-epoch lists.
        trajectories (dict): Per-evaluation-epoch trajectories (if logged).
        _losses_and_r2s (dict): Per-evaluation-epoch losses and r2 values
            of each test experiment (if evaluation done). For debugging only.
    """

    num_exps = len(experiments)

    # Presplit keys for initialization
    train_keys = jax.random.split(key, num_epochs * num_exps
                                  ).reshape((num_epochs, num_exps) + key.shape)

    # Initialize trainable weights by copying corresponding layers of exp.w_init_train
    params['w_init_learned'] = []
    for exp in experiments:
        params['w_init_learned'].append(
            {layer: exp.w_init_train[layer]
             for layer in cfg.training.trainable_init_weights})

    opt_state = optimizer.init(params)

    expdata = {}  # Per-metric dict of per-evaluation-epoch lists
    _losses_and_r2s = {}  # Per-evaluation-epoch losses and r2 values (debugging only)
    trajectories = {}  # Per-evaluation-epoch trajectories (debugging only)

    for epoch in range(num_epochs):
        epoch_keys = train_keys[epoch]
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
                epoch_trajectories.append(aux.pop('trajectories'))

            epoch_losses.append(aux)

        # For initial weights learning of test experiments in evaluation
        if not do_log:
            return params

        if epoch % cfg.logging.log_interval == 0:
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

