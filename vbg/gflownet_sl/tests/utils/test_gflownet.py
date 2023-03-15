import pytest

import numpy as np
import jax.numpy as jnp
import haiku as hk

from jax import random
from gflownet_sl.utils.gflownet import GFlowNetOutput, log_policy, uniform_log_policy


def test_log_policy():
    rng = hk.PRNGSequence(1357)
    outputs = GFlowNetOutput(
        logits=random.normal(next(rng), shape=(3, 5)),
        stop=random.normal(next(rng), shape=(3, 1))
    )
    masks = jnp.array([
        [0., 1., 1., 0., 1.],
        [1., 1., 0., 1., 1.],
        [0., 0., 0., 1., 0.]
    ])
    log_probas = log_policy(outputs, masks)
    probas = jnp.exp(log_probas)

    assert log_probas.shape == (3, 6)
    assert np.all(0. <= probas)
    assert np.all(probas <= 1.)
    np.testing.assert_allclose(jnp.sum(probas, axis=1), 1.)

    probas_continue = probas[:, :-1]
    np.testing.assert_array_equal(probas_continue[masks == 0.], 0.)


def test_log_policy_no_continue():
    rng = hk.PRNGSequence(1357)
    outputs = GFlowNetOutput(
        logits=random.normal(next(rng), shape=(1, 5)),
        stop=random.normal(next(rng), shape=(1, 1))
    )
    masks = jnp.array([[0., 0., 0., 0., 0.]])
    log_probas = log_policy(outputs, masks)
    probas = jnp.exp(log_probas)

    assert log_probas.shape == (1, 6)
    assert np.all(0. <= probas)
    assert np.all(probas <= 1.)
    np.testing.assert_allclose(jnp.sum(probas, axis=1), 1.)

    np.testing.assert_array_equal(probas[:, :-1], 0.)
    np.testing.assert_array_equal(probas[:, -1], 1.)


def test_uniform_log_policy():
    masks = jnp.array([
        [0., 1., 1., 0., 1.],
        [1., 1., 0., 1., 1.],
        [0., 0., 0., 1., 0.]
    ])
    log_probas = uniform_log_policy(masks)
    probas = jnp.exp(log_probas)

    assert log_probas.shape == (3, 6)
    expected_probas = np.array([
        [ 0., 1/4, 1/4,  0., 1/4, 1/4],
        [1/5, 1/5,  0., 1/5, 1/5, 1/5],
        [ 0.,  0.,  0., 1/2,  0., 1/2]
    ], dtype=np.float32)
    np.testing.assert_allclose(probas, expected_probas)


def test_uniform_log_policy_no_continue():
    masks = jnp.array([[0., 0., 0., 0., 0.]])
    log_probas = uniform_log_policy(masks)
    probas = jnp.exp(log_probas)

    assert log_probas.shape == (1, 6)
    expected_probas = np.array([[0., 0., 0., 0., 0., 1.]], dtype=np.float32)
    np.testing.assert_allclose(probas, expected_probas)
