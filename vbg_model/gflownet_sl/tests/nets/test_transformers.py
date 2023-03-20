import pytest

import numpy as np
import jax.numpy as jnp
import haiku as hk

from jax import random
from gflownet_sl.nets.transformers import DenseBlock, TransformerBlock


def test_dense_block():
    @hk.without_apply_rng
    @hk.transform
    def model(inputs):
        return DenseBlock(
            output_size=7,
            init_scale=1.,
            widening_factor=2
        )(inputs)

    rng = hk.PRNGSequence(1357)
    inputs = random.normal(next(rng), shape=(3, 5))
    params = model.init(next(rng), inputs)
    outputs = model.apply(params, inputs)

    assert outputs.shape == (3, 7)


def test_transformer_block():
    @hk.transform
    def model(hiddens, inputs, is_training):
        return TransformerBlock(
            num_heads=3,
            key_size=5,
            embedding_size=7,
            init_scale=1.,
            dropout_rate=0.1,
            widening_factor=2
        )(hiddens, inputs, is_training)

    rng = hk.PRNGSequence(1357)
    inputs = random.normal(next(rng), shape=(3, 1))
    hiddens = random.normal(next(rng), shape=(3, 3 * 5))
    params = model.init(next(rng), hiddens, inputs, True)
    outputs = model.apply(params, next(rng), hiddens, inputs, True)

    assert outputs.shape == hiddens.shape
