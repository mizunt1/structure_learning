import pytest

import numpy as np
import jax.numpy as jnp
import haiku as hk

from jax import random, vmap
from gflownet_sl.nets.gflownet import gflownet
from gflownet_sl.utils.gflownet import GFlowNetOutput


def test_gflownet():
    rng = hk.PRNGSequence(1357)
    model = hk.transform(gflownet)

    adjacency = random.bernoulli(next(rng), 0.5, shape=(3, 3))
    adjacency = adjacency.astype(jnp.float32)

    params = model.init(next(rng), adjacency, True)
    outputs = model.apply(params, next(rng), adjacency, True)

    assert isinstance(outputs, GFlowNetOutput)
    assert outputs.logits.shape == (9,)
    assert outputs.stop.shape == (1,)


def test_gflownet_vmap():
    rng = hk.PRNGSequence(1357)
    model = hk.transform(gflownet)

    adjacency = random.bernoulli(next(rng), 0.5, shape=(5, 3, 3))
    adjacency = adjacency.astype(jnp.float32)

    params = model.init(next(rng), adjacency[0], True)
    subkeys = jnp.array(rng.take(adjacency.shape[0]))
    outputs = vmap(model.apply, in_axes=(None, 0, 0, None))(params, subkeys, adjacency, True)

    assert isinstance(outputs, GFlowNetOutput)
    assert outputs.logits.shape == (5, 9)
    assert outputs.stop.shape == (5, 1)
