
import jax
import jax.numpy as jnp
import losses
import optax


def evaluate(key, cfg, theta, plasticity_func,
             train_experiments, init_trainable_weights_train,
             test_experiments, init_trainable_weights_test,
             expdata):

    # Evaluate train loss
    key, train_losses, _, _, _ = evaluate_loss(key,  # TODO key
                                          cfg,
                                          train_experiments,
                                          theta, plasticity_func,
                                          init_trainable_weights_train
    )

    # Learn initial weights for test experiments
    learned_init_trainable_weights_test = learn_initial_weights(
        key, cfg, theta, plasticity_func,  # TODO key
        test_experiments, init_trainable_weights_test)

    # Simulate test experiments with learned weights and plasticity


    # Evaluate test loss


    # Evaluate percent deviance explained


    # Evaluate R2

    return expdata

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

def evaluate_loss(key, cfg, experiments, theta, plasticity_func,
                  init_trainable_weights):

    total_losses, neural_losses, behavioral_losses, trajectories = [], [], [], []
    for exp in experiments:
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

        total_losses.append(loss)
        neural_losses.append(aux['neural'])
        behavioral_losses.append(aux['behavioral'])
        trajectories.append(aux['trajectories'])

    return (key, jnp.array(total_losses),
            jnp.array(neural_losses), jnp.array(behavioral_losses),
            trajectories)
