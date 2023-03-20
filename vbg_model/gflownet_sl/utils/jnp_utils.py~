import jax.numpy as jnp
import optax

from jax import random, jit, tree_util
from gflownet_sl.utils.gflownet import uniform_log_policy


def batch_random_choice(key, probas, masks):
    # Sample from the distribution
    uniform = random.uniform(key, shape=(probas.shape[0], 1))
    cum_probas = jnp.cumsum(probas, axis=1)
    samples = jnp.sum(cum_probas < uniform, axis=1, keepdims=True)

    # In rare case, the sampled actions may be invalid, despite having probability 0.
    # In those case, we select the stop action by default.
    stop_mask = jnp.ones((masks.shape[0], 1), dtype=masks.dtype)  # Stop action is always valid
    masks = masks.reshape(masks.shape[0], -1)
    masks = jnp.concatenate((masks, stop_mask), axis=1)

    is_valid = jnp.take_along_axis(masks, samples, axis=1)
    stop_action = masks.shape[1]
    samples = jnp.where(is_valid, samples, stop_action)

    return jnp.squeeze(samples, axis=1)

@jit
def get_random_actions(state, masks):
    key, subkey = random.split(state.key)
    log_probas = uniform_log_policy(masks.astype(jnp.float32))
    actions = batch_random_choice(subkey, jnp.exp(log_probas), masks)
    return (actions, state._replace(key=key))


@jit
def tree_mse(tree1, tree2):
    norm_diffs = tree_util.tree_map(optax.l2_loss, tree1, tree2)
    leaves = tree_util.tree_leaves(norm_diffs)

    size = sum(leaf.size for leaf in leaves)
    return sum(jnp.sum(leaf) for leaf in leaves) / size
