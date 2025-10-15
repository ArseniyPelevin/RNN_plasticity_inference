from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("returns"))
def simulate_step(step_variables, plasticity, returns):
    """ Simulate network activity and outputs in one time step.

    Args:
        step_variables: (
            network: Network,
            y_old (N_y_neurons,): y at previous step,
            x_input (N_x_neurons,): input at this step (without noise added),
            rewarded_pos (bool): whether this step is rewarded if licked,
            valid (bool): whether this step is real (not padding),
            step_key
        )
        plasticity: dict of plasticity modules for each plastic layer.
        returns: Tuple of strings indicating which outputs to return.

    Returns:
        (network: Updated network.
         y_new (N_y_neurons,): y after this step
        )
        output_data: dict with keys depending on `returns`:
            {xs, - (N_x_neurons,) for debugging
             xs, - (N_x_neurons,) for debugging
             ys, - (N_y_neurons,) in (0, 1)
             outputs, - (1,) pre-sigmoid logit (float)
             decisions, - (1,) boolean decision
             rewards, - (1,) boolean reward
             weights - (dict of plastic layers)
             }
    """
    network, *step_variables = step_variables

    network, *activity = network(step_variables, plasticity)

    # Construct dictionary of step outputs asked for return.
    # Will be stacked to form trajectory in scans
    output_data = {}
    for i, name in enumerate(('xs', 'ys', 'outputs', 'decisions', 'rewards')):
        if name in returns:
            output_data[name] = activity[i]

    if 'weights' in returns:
        output_data['weights'] = {}
        # Only return trajectories of plastic weights (from updated weights)
        if "ff" in network.cfg.plasticity_layers:
            output_data['weights']['w_ff'] = network.ff_layer.weight
        if "rec" in network.cfg.plasticity_layers:
            output_data['weights']['w_rec'] = network.rec_layer.weight
            output_data['weights']['b_rec'] = network.rec_layer.bias

    return (network, activity[1]), output_data

@partial(jax.jit, static_argnames=("returns"))
def simulate_trajectory(
    key,
    exp,
    x_input,
    network,
    plasticity,
    returns
    ):
    """ Simulate trajectory of activations (and weights) of one experiment (animal).

    Args:
        key: Random key for PRNG.
        exp: Experiment object with experiment parameters.
            # x_input and network are passed separately from exp,
            # because they are different in generation.
        x_input (N_sessions, N_steps_per_session_max, N_x_neurons): Input without noise.
        network: Initialized Network object with initial weights.
        returns: Tuple of strings indicating which trajectories to return.

    Returns:
        activity_trajec_exp: dict with keys depending on `returns`:
        {
            y_trajec_exp (N_sessions, N_steps_per_session_max, N_y_neurons),
            output_trajec_exp (N_sessions, N_steps_per_session_max),
            decision_trajec_exp (N_sessions, N_steps_per_session_max),
            xs_trajec_exp (N_sessions, N_steps_per_session_max, N_x_neurons),
            rewards_trajec_exp (N_sessions, N_steps_per_session_max),
            weights_trajec_exp {  # Only the plastic layers
                w_ff: (N_sessions, N_steps_per_session_max,
                       N_x_neurons, N_y_neurons),
                w_rec: (N_sessions, N_steps_per_session_max,
                        N_y_neurons, N_y_neurons)
                }
        }: Trajectories of activations over the course of the experiment.
    """

    # Pre-split keys for each session and step
    y_key, exp_key = jax.random.split(key)
    n_sessions = exp.step_mask.shape[0]
    n_steps = exp.step_mask.shape[1]
    y_keys = jax.random.split(y_key, n_sessions)  # For initial y of each session
    flat_keys = jax.random.split(exp_key, n_sessions * n_steps * 2)  # Two keys per step
    exp_keys = flat_keys.reshape((n_sessions, n_steps, 2) + exp_key.shape)

    def simulate_session(network, session_variables):
        """ Simulate trajectory of weights and activations within one session.

        Args:
            weights: Initial weights at the start of the session.
            session_variables: Tuple of (
                session_x,
                session_rewarded_pos,
                session_mask,
                session_keys,
                y_keys) for the session.

        Returns:
            weights_session: Parameters at the end of the session.
            activity_trajec_session: {
                y_trajec_session (N_steps_per_session_max, N_y_neurons),
                output_trajec_session (N_steps_per_session_max),
                decision_trajec_session (N_steps_per_session_max),
                xs_trajec_session (N_steps_per_session_max, N_x_neurons),
                rewards_trajec_session (N_steps_per_session_max),
                weights_trajec_session: {  # Only the plastic layers
                    w_ff: (N_steps_per_session_max, N_x_neurons, N_y_neurons),
                    w_rec: (N_steps_per_session_max, N_y_neurons, N_y_neurons),
                }
            }
        """
        def _simulate_step(step_carry, step_variables):
            # step_carry: (network, y_old)
            # step_variables: (x_input, rewarded_pos, valid, step_keys)
            step_variables = (*step_carry, *step_variables)
            return simulate_step(step_variables, plasticity, returns)

        *session_variables, y_key = session_variables
        # Initialize y activity at start of session
        init_y = jax.random.normal(y_key, (network.cfg.num_y_neurons,))
        init_y = jax.nn.sigmoid(init_y)  # Initial activity between 0 and 1

        # Run inner scan over steps within one session
        (network, _), session_output = jax.lax.scan(
            _simulate_step, (network, init_y), session_variables)

        return network, session_output

    # Reset running averages in the network at the start of simulation
    network = eqx.tree_at(
        lambda n: (n.mean_y_activation, n.expected_reward),
        network,
        (jnp.zeros((network.cfg.num_y_neurons,)), 0.0),
        )

    # Run outer scan over sessions within one experiment
    _carry, activity_trajec_exp = jax.lax.scan(
        simulate_session,
        network,
        (x_input,
         exp.rewarded_pos,
         exp.step_mask,
         exp_keys,
         y_keys)
    )

    # Zero out padding steps in trajectories
    for name, traj in activity_trajec_exp.items():
        if name != 'weights':
            activity_trajec_exp[name] = (traj * exp.step_mask[..., None]
                                         if name in ['xs', 'ys']
                                         else traj * exp.step_mask)
    return activity_trajec_exp
