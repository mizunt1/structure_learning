import jax.numpy as jnp
import haiku as hk
import optax

from functools import partial
from collections import namedtuple
from jax import grad, jit, random, vmap

from gflownet_sl.nets.gflownet import gflownet
from gflownet_sl.utils.gflownet import (uniform_log_policy,
    detailed_balance_loss, log_policy)
from gflownet_sl.utils.jnp_utils import batch_random_choice, tree_mse
from gflownet_sl.utils.multistep import get_nstep_indices


GFlowNetState = namedtuple('GFlowNetState', ['optimizer', 'key'])


class GFlowNet:
    def __init__(self, optimizer, delta=1., n_step=1):
        self.model = hk.transform(gflownet)
        self.optimizer = optax.chain(optimizer, optax.zero_nans())
        self.delta = delta
        self.n_step = n_step

        self._target_params = None
        self._subsq = get_nstep_indices(n_step)

    def loss(self, params, target_params, key, samples, subsq_mask, tau, is_training):
        n_step, batch_size = samples['adjacency'].shape[:2]

        key, *subkeys_t = random.split(key, n_step * batch_size + 1)
        subkeys_t = jnp.asarray(subkeys_t).reshape(n_step, batch_size, -1)

        _, *subkeys_tp1 = random.split(key, n_step * batch_size + 1)
        subkeys_tp1 = jnp.asarray(subkeys_tp1).reshape(n_step, batch_size, -1)

        vmodel = vmap(vmap(self.model.apply,
            in_axes=(None, 0, 0, None)),
            in_axes=(None, 0, 0, None)
        )
        outputs_t = vmodel(params, subkeys_t, samples['adjacency'], is_training)
        outputs_tp1 = vmodel(target_params, subkeys_tp1, samples['next_adjacency'], False)

        log_pi_t = log_policy(outputs_t, samples['mask'])
        log_pi_tp1 = log_policy(outputs_tp1, samples['next_mask'])

        return detailed_balance_loss(log_pi_t, log_pi_tp1, samples['actions'],
            samples['rewards'] / tau, samples['num_edges'],
            self._subsq, subsq_mask, delta=self.delta)

    def set_target(self, params):
        self._target_params = params

    @property
    def target_params(self):
        return self._target_params

    @partial(jit, static_argnums=(0,))
    def act(self, params, state, observations, epsilon):
        masks = observations['mask'].astype(jnp.float32)
        adjacencies = observations['adjacency'].astype(jnp.float32)
        batch_size = adjacencies.shape[0]
        key, subkey1, subkey2, *subkeys = random.split(state.key, batch_size + 3)

        # Get policy
        vmodel = vmap(self.model.apply, in_axes=(None, 0, 0, None))
        outputs = vmodel(params, jnp.asarray(subkeys), adjacencies, False)
        log_pi = log_policy(outputs, masks)

        # Get uniform policy
        log_uniform = uniform_log_policy(masks)

        # Mixture of policy and uniform
        is_exploration = random.bernoulli(subkey1, p=1. - epsilon, shape=(batch_size, 1))
        log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)

        state = state._replace(key=key)
        return actions, is_exploration.astype(jnp.int32), state

    @partial(jit, static_argnums=(0,))
    def step(self, params, target_params, state, samples, subsq, tau):
        key, subkey = random.split(state.key)
        grads, logs = grad(self.loss, has_aux=True)(
            params, target_params, subkey, samples, subsq, tau, True)

        # Update the params
        updates, state_opt = self.optimizer.update(grads, state.optimizer, params)
        params = optax.apply_updates(params, updates)
        state = GFlowNetState(optimizer=state_opt, key=key)

        # Add MSE between the parameters of the online & target networks to the logs
        logs['target/mse'] = tree_mse(params, target_params)

        return (params, state, logs)

    def init(self, key, adjacency):
        key, subkey = random.split(key)
        params = self.model.init(subkey, adjacency, True)
        self.set_target(params)  # Set target network parameters
        state = GFlowNetState(optimizer=self.optimizer.init(params), key=key)
        return (params, state)

    def reset_optim(self, state, params, key):
        state = GFlowNetState(optimizer=self.optimizer.init(params), key=key)
        return state
