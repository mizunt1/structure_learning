import jax.numpy as jnp
import haiku as hk

from jax import nn

from gflownet_sl.nets.utils.fast_attention import make_fast_softmax_attention


class MultiHeadDotProductFastAttention(hk.MultiHeadAttention):
    def __call__(self, query, key, value, mask=None):
        query_heads = self._linear_projection(query, self.key_size, 'query')
        key_heads = self._linear_projection(key, self.key_size, 'key')
        value_heads = self._linear_projection(value, self.value_size, 'value')

        attention_fn = make_fast_softmax_attention(
            self.key_size,
            lax_scan_unroll=16,
            nb_features=256,
            redraw_features=True
        )

        attn = attention_fn(query_heads, key_heads, value_heads)
        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))
        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)


class LinearMultiHeadAttention(hk.MultiHeadAttention):
    def __call__(self, query, key, value, mask=None):
        feature_map = lambda x: nn.elu(x) + 1.
        eps = 1e-6

        query_heads = self._linear_projection(query, self.key_size, 'query')
        key_heads = self._linear_projection(key, self.key_size, 'key')
        value_heads = self._linear_projection(value, self.value_size, 'value')

        # Map the query & key with a feature map
        query_heads = feature_map(query_heads)
        key_heads = feature_map(key_heads)

        key_values = jnp.einsum('...thd,...thk->...hkd', key_heads, value_heads)
        normalizer = 1. / (jnp.einsum('...thd,...hd->...th',
            query_heads, jnp.sum(key_heads, axis=-3)) + eps)
        attn = jnp.einsum('...thd,...hkd,...th->...thk',
            query_heads, key_values, normalizer)

        # Concatenate attention matrix of all heads into a single vector.
        attn_vec = jnp.reshape(attn, (*query.shape[:-1], -1))
        return hk.Linear(self.model_size, w_init=self.w_init)(attn_vec)


if __name__ == '__main__':
    from jax import random

    @hk.without_apply_rng
    @hk.transform
    def model(inputs):
        return LinearMultiHeadAttention(
            num_heads=2,
            key_size=3,
            w_init_scale=1.
        )(inputs, inputs, inputs)

    rng = hk.PRNGSequence(0)
    inputs = random.normal(next(rng), shape=(5, 7, 2 * 3))
    params = model.init(next(rng), inputs)
    outputs = model.apply(params, inputs)
    print(outputs.shape)  # (5, 7, 6)
