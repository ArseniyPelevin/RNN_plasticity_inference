# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "equinox",
#     "jax",
#     "omegaconf",
#     "pandas",
#     "optax",
#     "h5py",
# ]
# ///
import time

import equinox as eqx
import experiment
import jax
import jax.numpy as jnp
import losses

# from model_Jan_2 import VolterraPlasticity, Network
# from network import Network
# import plasticity
import main
import optax
import plasticity
import simulation
import training

# @partial(eqx.filter_jit)
# def simulation_step(network, step_variables, plasticity):
#     return network(step_variables, plasticity)


# @partial(eqx.filter_jit)
# def simulate(network, prev_y, variables):
#     # this is reusing the same inputs and rewards to be comparable to the other
#     # functions, in production we should scan over different inputs/rewards per
#     # timestep ("_" argument becomes "(inputs, reward)")
#     def outer_scan_function(carry, step_variables):
#         def scan_function(carry, step_variables):
#             network, prev_y = carry
#             keys = jax.random.split(jax.random.PRNGKey(0), 2)  # dummy keys
#             step_variables = (prev_y, *step_variables, keys)  # replace prev_y in step_variables
#             network, *outputs = network(step_variables, plasticity)
#             return (network, outputs[1]), outputs
#         return jax.lax.scan(scan_function, carry, step_variables)

#     (network, y), outputs = jax.lax.scan(
#         outer_scan_function, (network, prev_y), variables
#     )

#     return network, y, outputs

def prepare_for_meta_learning_step(key, cfg, train_experiments):
    plasticity_train = plasticity.initialize_plasticity(
        key, cfg.plasticity, mode='training')

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

    return params, plasticity_train, optimizer, opt_state


if __name__ == "__main__":

    key = jax.random.PRNGKey(1912)
    keys = jax.random.split(key, 2)

    num_iter = 100

    cfg = main.create_config()
    cfg = main.validate_config(cfg)
    cfg.network.num_x_neurons = 100
    cfg.network.num_y_neurons = 500
    cfg.experiment.mean_steps_per_trial = num_iter

    cfg.experiment.num_exp_train = 3
    cfg.experiment.num_exp_test = 3
    cfg.training.num_epochs = 9
    cfg.training.num_epochs_weights = 3

    cfg.logging.log_trajectories = False
    train_experiments = experiment.generate_experiments(key, cfg,
                                                        mode='train')
    test_experiments = experiment.generate_experiments(key, cfg,
                                                       mode='test')

    exp = train_experiments[0]
    inputs = exp.x_input
    mask = exp.step_mask
    reward = exp.rewarded_pos
    network = exp.network

    prev_y = jnp.zeros((network.cfg.num_y_neurons,), dtype=jnp.float32)

    variables = (inputs, reward, mask)
    step_variables = (prev_y, inputs[0][0], reward[0][0], mask[0][0], keys)

    params, plasticity_train, optimizer, opt_state = prepare_for_meta_learning_step(
        keys[0], cfg, train_experiments)

    loss_value_and_grad = eqx.filter_value_and_grad(losses.loss, has_aux=True)
    #_________________________________________________
    # warmup
    # network, *outputs = network(step_variables)
    # outputs[1].block_until_ready()
    # network, *outputs = simulation_step(network, step_variables)
    # outputs[1].block_until_ready()
    out = simulation.simulate_step(
            network, step_variables, plasticity_train, returns=('ys', 'outputs')
        )
    jax.device_get(out)
    print("Finished simulate_step warmup")
    # network, prev_y, *outputs = simulate(
    #     network,
    #     prev_y,
    #     variables,
    # )
    # outputs[0][1].block_until_ready()
    out = simulation.simulate_trajectory(
        key,
        exp,
        inputs,
        network,
        plasticity_train,
        returns=('ys', 'outputs')
    )
    jax.device_get(out)
    print("Finished simulate_trajectory warmup")

    out = losses.loss(params, key, exp, plasticity_train, cfg, returns=())
    jax.device_get(out)
    print("Finished loss warmup")

    out = loss_value_and_grad(params, key, exp, plasticity_train,
                                             cfg, returns=())
    jax.device_get(out)
    print("Finished loss + grad warmup")

    out = training.meta_learning_step(
        params, key, exp, plasticity_train, cfg, optimizer, opt_state)
    jax.device_get(out)
    print("Finished meta_learning_step warmup")

    cfg.logging.do_evaluation = False
    out = training.meta_learn_plasticity(key, cfg,
                                                           train_experiments,
                                                           test_experiments)
    jax.device_get(out)
    print("Finished training without evaluation warmup")

    cfg.logging.do_evaluation = True
    out  = training.meta_learn_plasticity(key, cfg,
                                                            train_experiments,
                                                            test_experiments)
    jax.device_get(out)
    print("Finished training with evaluation warmup")

    #_________________________________________________


    print("FULL PLASTICITY, FULL TRAINABLE WEIGHTS, SAME THETAS:")
    # # single steps, no JIT
    # start = time.time()
    # num_timesteps = 100
    # for i in range(num_timesteps):
    #     network, *outputs = network(step_variables)
    #     prev_y = outputs[0]
    # outputs[1].block_until_ready()
    # print(f"network forward: {(time.time() - start) / num_timesteps*1000:.3f}ms per iteration")

    # # single steps, JIT
    # start = time.time()
    # num_timesteps = 100
    # for i in range(num_timesteps):
    #     network, *outputs = simulation_step(network, step_variables)
    # outputs[1].block_until_ready()
    # print(f"JIT step: {(time.time() - start) / num_timesteps * 1000:.3f}ms per iteration")

    # my single step
    start = time.time()
    num_timesteps = num_iter
    for i in range(num_timesteps):
        out = simulation.simulate_step(
            network, step_variables, plasticity_train, returns=('ys', 'outputs')
        )
        jax.device_get(out)
    print("simulate_step:",
          f"{(time.time() - start) / num_timesteps * 1000:.3f}ms per iteration")

    # # multiple steps, JIT
    # num_timesteps = exp.cfg.mean_steps_per_trial
    # start = time.time()
    # network, prev_y, *outputs = simulate(
    #     network,
    #     prev_y,
    #     variables,
    # )
    # outputs[0][1].block_until_ready()
    # print(
    #     f"JIT+scan: {(time.time() - start) / num_timesteps * 1000:.3f}ms per iteration"
    # )

    # simulate trajectory
    num_timesteps = exp.cfg.mean_steps_per_trial * cfg.experiment.num_exp_train
    start = time.time()
    for exp in train_experiments:
        out = simulation.simulate_trajectory(
            key,
            exp,
            inputs,
            network,
            plasticity_train,
            returns=('ys', 'outputs')
        )
        jax.device_get(out)
    print(
        "simulate_trajectory:",
        f"{(time.time() - start) / num_timesteps * 1000:.3f}ms per iteration"
    )

    # compute loss
    num_timesteps = exp.cfg.mean_steps_per_trial * cfg.experiment.num_exp_train
    start = time.time()
    for exp in train_experiments:
        out = losses.loss(params, key, exp, plasticity_train, cfg, returns=())
        jax.device_get(out)
    full_time = time.time() - start
    print(f"loss: {full_time:.3f}s, ",
          f"{full_time / num_timesteps * 1000:.3f}ms per iteration")

    # compute loss + grad
    num_timesteps = exp.cfg.mean_steps_per_trial * cfg.experiment.num_exp_train
    start = time.time()
    for exp in train_experiments:
        out = loss_value_and_grad(params, key, exp, plasticity_train,
                                   cfg, returns=())
        jax.device_get(out)
    full_time = time.time() - start
    print("loss + grad:",
          f"Full time: {full_time:.3f}s, ",
          f"{full_time / num_timesteps * 1000:.3f}ms per iteration")

    # meta-learning step
    num_timesteps = (exp.cfg.mean_steps_per_trial *
                     cfg.experiment.num_exp_train)
    start = time.time()
    for exp in train_experiments:
        out = training.meta_learning_step(
            params, key, exp, plasticity_train, cfg, optimizer, opt_state)
        jax.device_get(out)
    full_time = time.time() - start
    print("meta_learning_step:",
          f"Full time: {full_time:.3f}s, ",
          f"{full_time / num_timesteps * 1000:.3f}ms per iteration"
    )

    # full meta training loop without evaluation
    num_timesteps = (exp.cfg.mean_steps_per_trial *
                     (cfg.training.num_epochs + 1) *  # code adds +1 internally
                     cfg.experiment.num_exp_train)
    cfg.logging.do_evaluation = False
    start = time.time()
    out = training.meta_learn_plasticity(key, cfg,
                                                     train_experiments,
                                                     test_experiments)
    jax.device_get(out)
    full_time = time.time() - start
    print("Training without evaluation:",
          f"Full time: {full_time:.3f}s, ",
          f"{full_time / num_timesteps * 1000:.3f}ms per iteration"
    )

    # full meta training loop with evaluation
    num_timesteps = (exp.cfg.mean_steps_per_trial *
                     (cfg.training.num_epochs + 1) *
                     cfg.experiment.num_exp_train + (cfg.experiment.num_exp_test *
                                                     cfg.training.num_epochs_weights /
                                                     cfg.logging.log_interval))
    cfg.logging.do_evaluation = True
    start = time.time()
    out = training.meta_learn_plasticity(key, cfg,
                                                     train_experiments,
                                                     test_experiments)
    jax.device_get(out)
    full_time = time.time() - start
    print("Training with evaluation:",
          f"Full time: {full_time:.3f}s, ",
          f"{full_time / num_timesteps * 1000:.3f}ms per iteration"
    )

    # print(f"{prev_y[:100]=}")
    # print(f"{network.mean_y_activation[:100]=}")
    # print(f"{network.rec_layer.weight[:, 0]=}")
    # print(f"{network.rec_layer.bias[:100]=}")
