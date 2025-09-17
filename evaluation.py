
import jax
import jax.numpy as jnp
import losses
import optax
import synapse


def evaluate(key, cfg, theta, plasticity_func,
             train_experiments, init_trainable_weights_train,
             test_experiments, init_trainable_weights_test,
             expdata):

    # Evaluate train loss
    train_losses, _, _, _ = evaluate_loss(key,  # TODO key
                                          cfg,
                                          train_experiments,
                                          plasticity_func,
                                          theta,
                                          init_trainable_weights_train
    )

    # Learn initial weights for test experiments
    learned_init_weights = learn_initial_weights(
        key, cfg, theta, plasticity_func,  # TODO key
        test_experiments, init_trainable_weights_test)

    zero_theta, _ = synapse.init_plasticity_volterra(key=None, init="zeros", scale=None)

    # Compute loss of full model with learned plasticity and learned weights
    test_losses, MSE_F, BCE_F, activation_trajs_F = evaluate_loss(key, cfg,
                                                        test_experiments,
                                                        plasticity_func,
                                                        theta, learned_init_weights
                                                        )
    print(f"MSE_F={[f'{float(v):.6f}'  for v in jnp.asarray(MSE_F).tolist()]}")

    # Compute loss of theta model with learned plasticity and random weights
    _, MSE_T, BCE_T, activation_trajs_T = evaluate_loss(key, cfg,
                                                        test_experiments,
                                                        plasticity_func,
                                                        theta, init_trainable_weights_train
                                                        )
    print(f"MSE_T={[f'{float(v):.6f}' for v in jnp.asarray(MSE_T).tolist()]}")


    # Compute loss of weights model with zero plasticity and learned weights
    _, MSE_W, BCE_W, activation_trajs_W = evaluate_loss(key, cfg,
                                                        test_experiments,
                                                        plasticity_func,
                                                        zero_theta, learned_init_weights
                                                        )
    print(f"MSE_W={[f'{float(v):.6f}' for v in jnp.asarray(MSE_W).tolist()]}")


    # Compute loss of null model with zero plasticity and random weights
    _, MSE_N, BCE_N, activation_trajs_N = evaluate_loss(key, cfg,
                                                        test_experiments,
                                                        plasticity_func,
                                                        zero_theta, init_trainable_weights_train
                                                        )
    print(f"MSE_N={[f'{float(v):.6f}' for v in jnp.asarray(MSE_N).tolist()]}")

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

def evaluate_loss(key, cfg, experiments, plasticity_func,
                  theta, init_trainable_weights):

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

    return (jnp.array(total_losses),
            jnp.array(neural_losses), jnp.array(behavioral_losses),
            trajectories)
