import jax.numpy as jnp
import haiku as hk

from jax import nn

from gflownet_sl.nets.attention import (
    MultiHeadDotProductFastAttention, LinearMultiHeadAttention)


class DenseBlock(hk.Module):
    def __init__(self, output_size, init_scale, widening_factor=4, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.init_scale = init_scale
        self.widening_factor = widening_factor

    def __call__(self, inputs):
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        hiddens = hk.Linear(
            self.widening_factor * self.output_size,
            w_init=w_init
        )(inputs)
        hiddens = nn.gelu(hiddens)
        return hk.Linear(self.output_size, w_init=w_init)(hiddens)


class TransformerBlock(hk.Module):
    def __init__(
            self,
            num_heads,
            key_size,
            embedding_size,
            init_scale,
            dropout_rate,
            widening_factor=4,
            performer=False,
            linear=False,
            name=None
        ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.embedding_size = embedding_size
        self.init_scale = init_scale
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor
        self.performer = performer
        self.linear = linear

    def __call__(self, hiddens, inputs, is_training):
        dropout_rate = self.dropout_rate if is_training else 0.
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        if self.performer:
            attn_cls = MultiHeadDotProductFastAttention
        elif self.linear:
            attn_cls = LinearMultiHeadAttention
        else:
            attn_cls = hk.MultiHeadAttention

        inputs_embedding = hk.Linear(
            self.embedding_size,
            w_init=w_init,
            name='linear_1'
        )(inputs)
        h_norm = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name='layernorm_1'
        )(jnp.concatenate((inputs_embedding, hiddens), axis=-1))
        h_attn = attn_cls(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init_scale=self.init_scale
        )(h_norm, h_norm, h_norm, mask=None)
        h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
        hiddens = hiddens + h_attn

        inputs_embedding = hk.Linear(
            self.embedding_size,
            w_init=w_init,
            name='linear_2'
        )(inputs)
        h_norm = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name='layernorm_2'
        )(jnp.concatenate((inputs_embedding, hiddens), axis=-1))
        h_dense = DenseBlock(
            init_scale=self.init_scale,
            widening_factor=self.widening_factor,
            output_size=self.num_heads * self.key_size
        )(h_norm)
        h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
        hiddens = hiddens + h_dense

        return hiddens


if __name__ == '__main__':
    from jax import random

    @hk.transform
    def model(inputs, is_training):
        return TransformerBlock(4, 32, 2., 0.1)(inputs, is_training)

    rng = hk.PRNGSequence(0)
    inputs = random.normal(next(rng), shape=(3, 129))
    params = model.init(next(rng), inputs, True)
    outputs = model.apply(params, next(rng), inputs, True)

    print(inputs[:, :2])
    print(outputs[:, :2])
    print(outputs.shape)
