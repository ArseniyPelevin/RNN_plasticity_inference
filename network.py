import equinox as eqx
import jax
import jax.numpy as jnp


class Network(eqx.Module):
    ff_layer: eqx.Module
    rec_layer: eqx.Module
    out_layer: eqx.Module

    ff_mask: jnp.array
    rec_mask: jnp.array
    recording_mask: jnp.array
    ff_scale: jnp.array
    rec_scale: jnp.array

    mean_y_activation: jnp.array  # Running average of y activation for bias update
    expected_reward: float  # Running average of rewards when licked

    cfg: object = eqx.field(static=True)
    input_type: str = eqx.field(static=True)

    def __init__(self, key, cfg, input_type, mode):
        self.cfg = cfg
        self.input_type = input_type

        key1, key2, key3, mask_key = jax.random.split(key, 4)

        self.ff_layer = eqx.nn.Linear(self.cfg.num_x_neurons,
                                      self.cfg.num_y_neurons, key=key1)
        self.rec_layer = eqx.nn.Linear(self.cfg.num_y_neurons,
                                       self.cfg.num_y_neurons, key=key2)
        self.out_layer = eqx.nn.Linear(self.cfg.num_y_neurons,
                                       self.cfg.num_outputs, key=key3)

        self.mean_y_activation = jnp.zeros((self.cfg.num_y_neurons,))
        self.expected_reward = 0.0

        self.set_masks_and_scales(mask_key, mode)

    def set_masks_and_scales(self, key, mode):

        ff_mask_key, rec_mask_key, recording_mask_key = jax.random.split(key, 3)

        self.ff_mask = self.generate_feedforward_mask(
            ff_mask_key,
            self.cfg.num_x_neurons, self.cfg.num_y_neurons,
            self.cfg.feedforward_sparsity[mode],
            self.cfg.input_sparsity[mode]
        )
        self.rec_mask = self.generate_recurrent_mask(
            rec_mask_key,
            self.cfg.num_y_neurons, self.cfg.recurrent_sparsity[mode],
            self.ff_mask
        )
        self.recording_mask = jax.random.bernoulli(
            recording_mask_key,
            self.cfg.neural_recording_sparsity,
            self.cfg.num_y_neurons
        )
        self.ff_scale = self.compute_input_scale(self.ff_mask,
                                                 self.cfg.feedforward_input_scale)
        self.rec_scale = self.compute_input_scale(self.rec_mask,
                                                  self.cfg.recurrent_input_scale)

    def generate_feedforward_mask(self, key, n_pre, n_post,
                                  ff_sparsity, input_sparsity):
        """Generate a binary mask for the feedforward weights to enforce sparsity.

        Args:
            key: JAX random key.
            n_pre: Number of presynaptic neurons.
            n_post: Number of postsynaptic neurons.
            ff_sparsity [0, 1]: Fraction of nonzero weights in the feedforward layer,
                of all postsynaptic neurons receiving input (input_sparsity),
                all presynaptic neurons are guaranteed to have some output:
                0 - max(n_pre, n_post * input_sparsity) nonzero weights,
                1 - all presynaptic are connected to all input-receiving postsynaptic.
            input_sparsity [0, 1]: Fraction of postsynaptic neurons receiving input.

        Returns:
            A binary mask of shape (n_pre, n_post).
        """
        col_key, mask_key, fill_col_key, fill_row_key = jax.random.split(key, 4)

        # Choose input postsynaptic neurons (input columns)
        n_input_post = max(1, int(round(input_sparsity * n_post)))
        input_cols = jax.random.choice(
            col_key, n_post, shape=(n_input_post,), replace=False
        )

        # Generate random mask with given sparsity
        mask = jnp.zeros((n_pre, n_post))
        bern = jax.random.bernoulli(mask_key, p=float(ff_sparsity),
                                    shape=(n_pre, n_input_post))
        mask = mask.at[:, input_cols].set(bern)

        # Ensure no zero rows (presynaptic neurons without output)
        row_sums = mask[:, input_cols].sum(axis=1)
        zero_rows = jnp.where(row_sums == 0)[0]
        chosen_cols = jax.random.choice(
            fill_col_key, input_cols, shape=zero_rows.shape, replace=True
        )
        mask = mask.at[zero_rows, chosen_cols].set(1)

        # Ensure no zero selected columns (input postsynaptic neurons without input)
        col_sums = mask[:, input_cols].sum(axis=0)
        zero_cols_idx = jnp.where(col_sums == 0)[0]
        zero_cols = input_cols[zero_cols_idx]
        chosen_rows = jax.random.choice(
            fill_row_key, n_pre, shape=zero_cols.shape, replace=True
        )
        mask = mask.at[chosen_rows, zero_cols].set(1)

        return mask.astype(jnp.bool)

    def generate_recurrent_mask(self, key, n_post, rec_sparsity, ff_mask):
        """Generate a binary mask for the recurrent weights to enforce sparsity.

        Args:
            key: JAX random key.
            n_post: Number of postsynaptic neurons.
            rec_sparsity [0, 1]: Fraction of nonzero weights in the recurrent layer,
                all neurons are guaranteed to receive some input and some output:
                0 - at least one input per neuron,
                    not counting (allowed) autapses, but counting feedforward inputs,
                1 - all-to-all connectivity.
            ff_mask: Feedforward mask (n_pre, n_post) to count feedforward inputs.

        Returns:
            A binary mask of shape (n_post, n_post).
        """
        mask_key, fill_key = jax.random.split(key)

        # Generate random mask with given sparsity
        mask = jax.random.bernoulli(mask_key, p=float(rec_sparsity),
                                    shape=(n_post, n_post)
                                    ).astype(jnp.float32)

        # Construct test mask to ensure at least one input per neuron
        # Autapses are not counted as input
        test_mask = mask.at[jnp.diag_indices(n_post)].set(0)
        # Feedforward input is counted as input
        test_mask = jnp.vstack([test_mask, ff_mask.sum(axis=0)])

        # Ensure no zero columns in test mask (postsynaptic neurons without any input)
        col_sums = test_mask.sum(axis=0)
        zero_cols = jnp.where(col_sums == 0)[0]
        chosen_rows = jax.random.choice(
            fill_key, n_post, shape=zero_cols.shape, replace=True
        )
        # If any diagonal elements were chosen, shift them up by one
        diagonal = jnp.where(zero_cols == chosen_rows)[0]
        chosen_rows = chosen_rows.at[diagonal].subtract(1)

        mask = mask.at[chosen_rows, zero_cols].set(1)

        return mask.astype(jnp.bool)

    def compute_input_scale(self, mask, base_scale):
        """ Scale weights by constant and by number of inputs to each neuron."""
        n_inputs = mask.sum(axis=0) # N_inputs per postsynaptic neuron
        n_inputs = jnp.where(n_inputs == 0, 1, n_inputs) # avoid /0
        return base_scale / jnp.sqrt(n_inputs)[None, :]

    def apply_weights(self, weights):
        """ Set weights of the network to given values.

        Args:
            weights: Per-layer dict of arrays of shape
                (num_x_neurons, num_y_neurons) for 'ff' weights,
                (num_y_neurons, num_y_neurons) for 'rec' weights,
                (num_y_neurons,) for 'rec' biases,
                (num_y_neurons,) for 'out' weights,
                (1,) for 'out' biases.
        Returns:
            network: Updated network with new weights.
        """
        return eqx.tree_at(
            lambda network: (
                network.ff_layer.weight,
                network.rec_layer.weight,
                network.rec_layer.bias,
                network.out_layer.weight,
                network.out_layer.bias
            ),
            self,
            (
                weights['ff']['w'],
                weights['rec']['w'],
                weights['rec']['b'],
                weights['out']['w'].squeeze(),
                weights['out']['b'].squeeze()
            )
        )

    def __call__(self, step_variables, plasticity):
        """ Forward pass through the network for one time step,
        with plasticity updates.

        Args:
            step_variables: (y_old, inputs, rewarded_pos, valid, keys)
                y_old (N_y_neurons,): y at previous step,
                inputs (N_x_neurons,): input at this step (without noise added),
                rewarded_pos (bool): whether this step is rewarded if licked,
                valid (bool): whether this step is real (not padding),
                keys: (input_noise_key, decision_key)
            plasticity: dict of plasticity modules for each plastic layer.
        Returns:
            network: Updated network after plasticity.
            x (N_x_neurons,): x at this step (with noise added),
            y (N_y_neurons,): y at this step,
            output (N_outputs,): output at this step (pre-sigmoid logit),
            decision (1,): binary decision at this step,
            reward (1,): binary reward at this step.
        """
        y_old, inputs, rewarded_pos, valid, keys = step_variables
        input_noise_key, decision_key = keys

        # Compute x activity
        # Add noise to inputs on each step
        input_noise = jax.random.normal(input_noise_key, (self.cfg.num_x_neurons,))
        x = inputs + input_noise * self.cfg.input_noise_std
        # Make x positive if task input
        if self.input_type == 'task':
            x = jnp.clip(x, min=0)

        # Feedforward layer: x -- w_ff --> y

        # Apply scale and sparsity mask to ff weights
        w_ff = self.ff_layer.weight.T * self.ff_scale * self.ff_mask
        b_ff = self.ff_layer.bias
        # Compute feedforward activation
        y_activation = x @ w_ff + b_ff

        # Recurrent layer (if present): y -- w_rec --> y

        # Apply scale and sparsity mask to rec weights
        w_rec = self.rec_layer.weight.T * self.rec_scale * self.rec_mask
        b_rec = self.rec_layer.bias
        # Add recurrent activation
        y_activation += y_old @ w_rec + b_rec

        # Update moving average of y activation
        mean_y_activation = 0.9 * self.mean_y_activation + 0.1 * y_activation

        # Apply nonlinearity
        y = jax.nn.sigmoid(y_activation)

        # Compute output as pre-sigmoid logit (1,) based on recurrent layer activity
        output = self.out_layer(y).squeeze()

        decision = self.compute_decision(decision_key, output)

        # Reward if licked at rewarded position
        reward = (decision * rewarded_pos).astype(jnp.bool)
        # Expected reward is moving average of recent rewards when licked
        expected_reward = ((1 - 0.1 * decision) * self.expected_reward +
                           (0.1 * decision) * reward)

        network = self.update_weights(x, y, y_old, decision, reward, expected_reward,
                                      mean_y_activation, valid, plasticity)

        return network, x, y, output, decision, reward

    def compute_decision(self, key, output):
        """ Make binary decision based on output (probability of decision).
        To lick or not to lick at this step.

        min_lick_probability encourages licking only for reinforcement learning.
        Set to zero for fitting.

        Args:
            key: JAX random key.
            output (1,): Pre-sigmoid logit (float) for decision probability.

        Returns:
            decision (float): Binary decision (0 or 1).
        """

        return jax.random.bernoulli(key,
                                    jnp.maximum(self.cfg.min_lick_probability,
                                                jax.nn.sigmoid(output)))

    def update_weights(self, x, y, y_old, decision, reward, expected_reward,
                       mean_y_activation, valid, plasticity):
        def plasticity_vmap(pre, post, weights, reward):
            def plasticity_vmap_pre(pre, post, weights, reward):
                # vmap over pre-synapses
                # pre: (num_pre,)
                # post: (,)
                # weights: (num_pre,)
                # reward: (,)
                return jax.vmap(plasticity_function, in_axes=(0, None, 0, None))(
                    pre, post, weights, reward
                )
            # vmap over post-synapses
            # pre: (num_pre,)
            # post: (num_post,)
            # weights: (num_post, num_pre)
            # reward: (,)
            return jax.vmap(
                plasticity_vmap_pre,
                in_axes=(None, 0, 0, None),
            )(pre, post, weights, reward)

        reward_term = (reward - self.expected_reward) * decision

        # plasticity update for weights from input -> recurrent
        if 'ff' in self.cfg.plasticity_layers:
            if 'both' in plasticity:
                plasticity_function = plasticity['both']
            else:
                plasticity_function = plasticity['ff']
            ff_weights = plasticity_vmap(x, y, self.ff_layer.weight, reward_term)
            tanh_scale = self.cfg.synaptic_weight_threshold
            ff_weights = jax.nn.tanh(ff_weights / tanh_scale) * tanh_scale
            # Do not update weights on padded steps
            ff_weights = jnp.where(valid, ff_weights, self.ff_layer.weight)
        else:
            ff_weights = self.ff_layer.weight

        # plasticity update for weights from recurrent -> recurrent and
        # bias update to maintain target activation of 0.0 (before
        # nonlinearity)
        if 'rec' in self.cfg.plasticity_layers:
            if 'both' in plasticity:
                plasticity_function = plasticity['both']
            else:
                plasticity_function = plasticity['rec']
            rec_weights = plasticity_vmap(y_old, y, self.rec_layer.weight, reward_term)
            tanh_scale = self.cfg.synaptic_weight_threshold
            rec_weights = jax.nn.tanh(rec_weights / tanh_scale) * tanh_scale
            rec_biases = (self.rec_layer.bias -
                          self.cfg.homeostasis_rate * mean_y_activation)
            rec_weights = jnp.where(valid, rec_weights, self.rec_layer.weight)
            rec_biases = jnp.where(valid, rec_biases, self.rec_layer.bias)
        else:
            rec_weights = self.rec_layer.weight
            rec_biases = self.rec_layer.bias

        # Do not update state variables on padded steps
        mean_y_activation = jnp.where(valid,
                                      mean_y_activation, self.mean_y_activation)
        expected_reward = jnp.where(valid,
                                    expected_reward, self.expected_reward)

        return eqx.tree_at(
            lambda network: (
                network.ff_layer.weight,
                network.rec_layer.weight,
                network.rec_layer.bias,
                network.mean_y_activation,
                network.expected_reward,
            ),
            self,
            (ff_weights, rec_weights, rec_biases,
             mean_y_activation, expected_reward))

def initialize_weights(key, cfg, layers=('ff', 'rec', 'out')):
    """ Initialize trainable initial weights for one experiment.

    Args:
        cfg (dict): Configuration dictionary with network parameters.
        layers (list): List of layers to initialize weights for.
    Returns:
        w_init (dict): Per-layer dict of weight arrays.
    """
    ff_w_key, rec_w_key, rec_b_key, out_w_key, out_b_key = jax.random.split(key, 5)

    w_init = {}
    # Initialize feedforward weights
    if 'ff' in layers:
        w_init['ff'] = {}
        w_init['ff']['w'] = jax.random.normal(ff_w_key,
            (cfg.num_y_neurons,
             cfg.num_x_neurons)
             ) * cfg.init_weights_std['training']['ff']

    if 'rec' in layers:
        w_init['rec'] = {}
        w_init['rec']['w'] = jax.random.normal(
            rec_w_key,
            (cfg.num_y_neurons,
             cfg.num_y_neurons)
             ) * cfg.init_weights_std['training']['rec']
        w_init['rec']['b'] = jax.random.normal(
            rec_b_key,
            (cfg.num_y_neurons,)
             ) * cfg.init_weights_std['training']['rec']

    if 'out' in layers:
        w_init['out'] = {}
        w_init['out']['w'] = jax.random.normal(
            out_w_key,
            (cfg.num_outputs,
             cfg.num_y_neurons)
             ) * cfg.init_weights_std['training']['out']
        w_init['out']['b'] = jax.random.normal(
            out_b_key,
            (cfg.num_outputs,)
             ) * cfg.init_weights_std['training']['out']

    return w_init

def initialize_exps_weights(key, num_exps, cfg, layers=('ff', 'rec', 'out')):
    """ Initialize trainable initial weights for all experiments.

    Args:
        num_exps (int): Number of experiments to initialize weights for.
        cfg (dict): Configuration dictionary with network parameters.
        layers (list): List of layers to initialize weights for.
    Returns:
        w_inits (list): List of weights dicts for all experiments.
    """
    keys = jax.random.split(key, num_exps)
    w_inits = []
    for i in range(num_exps):
        w_init = initialize_weights(keys[i], cfg, layers)
        w_inits.append(w_init)
    return w_inits
