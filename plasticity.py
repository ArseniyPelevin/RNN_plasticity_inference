import equinox as eqx
import jax
import jax.numpy as jnp


def initialize_plasticity(key, cfg, mode):
    plasticity_functions = {'volterra': VolterraPlasticity,
                            'mlp': MLPPlasticity}
    plasticity = {}
    for layer in cfg.plasticity_models:
        plasticity[layer] = plasticity_functions[
            cfg.plasticity_models[layer]](
                key,
                learning_rate=cfg.synapse_learning_rate[layer],
                init_scale=cfg.plasticity_coeffs_init_scale[layer],
                coeff_masks=jnp.array(cfg.coeff_masks[layer]),  # For VolterraPlasticity
                hidden_sizes=None)  #cfg.mlp_hidden_sizes[layer])  # For MLPPlasticity
        if mode == 'generation':
            if cfg.plasticity_models[layer] == 'mlp':
                raise NotImplementedError("How should MLP generation coeffs be set?")
            for generation_coeff in cfg.generation_plasticity[layer]:
                plasticity[layer] = plasticity[layer].set_coefficient(
                    **generation_coeff)

    return plasticity


class VolterraPlasticity(eqx.Module):
    coefficients: jnp.array
    learning_rate: float
    coeff_masks: jnp.array

    def __init__(self, key, learning_rate, init_scale,
                 coeff_masks, hidden_sizes=None):
        # Initialize coefficients
        self.learning_rate = learning_rate
        self.coefficients = jax.random.normal(key, (3, 3, 3, 3)) * init_scale
        self.coeff_masks = coeff_masks

    def set_coefficient(
        self,
        value=0.0,
        pre=0,
        post=0,
        weight=0,
        reward=0
    ):
        coefficients = self.coefficients.at[pre, post, weight, reward].set(value)
        return eqx.tree_at(lambda p: p.coefficients, self, coefficients)

    def __call__(self, pre, post, weight, reward):
        pre_powers = jnp.array([1.0, pre, pre**2])
        post_powers = jnp.array([1.0, post, post**2])
        weight_powers = jnp.array([1.0, weight, weight**2])
        reward_powers = jnp.array([1.0, reward, reward**2])
        terms = jnp.outer(
            pre_powers, jnp.outer(post_powers, jnp.outer(weight_powers, reward_powers))
        ).reshape(3, 3, 3, 3)
        weight_update = jnp.sum(self.coefficients * terms)

        return weight + self.learning_rate * weight_update


class MLPPlasticity(eqx.Module):
    mlp: eqx.Module
    learning_rate: float

    def __init__(self, key, learning_rate, init_scale,
                 coeff_masks=None, hidden_sizes=None):
        layers = []
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        for i in range(len(hidden_sizes) - 1):
            layer = eqx.nn.Linear(hidden_sizes[i], hidden_sizes[i+1], key=keys[i])
            layer = eqx.tree_at(lambda layer: (layer.weight, layer.bias),
                                layer,
                                (layer.weight * init_scale, layer.bias * init_scale))
            layers.append(layer)
            layers.append(jax.nn.leaky_relu)
        layers.append(eqx.nn.Linear(hidden_sizes[-1], 1, key=keys[-1]))
        self.mlp = eqx.nn.Sequential(layers)
        self.learning_rate = learning_rate

    def __call__(self, pre, post, weight, reward):
        inputs = jnp.stack([pre, post, weight, reward])
        weight_update = self.mlp(inputs).squeeze()
        return weight + self.learning_rate * weight_update
