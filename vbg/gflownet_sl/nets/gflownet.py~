import jax.numpy as jnp
import haiku as hk
import math

from gflownet_sl.utils.gflownet import GFlowNetOutput
from gflownet_sl.nets.transformers import TransformerBlock


def gflownet(adjacency, is_training):
    # Create the edges as pairs of indices (source, target)
    num_variables = adjacency.shape[0]
    indices = jnp.arange(num_variables ** 2)
    sources, targets = jnp.divmod(indices, num_variables)
    edges = jnp.stack((sources, num_variables + targets), axis=1)

    # Embedding of the edges
    embeddings = hk.Embed(2 * num_variables, embed_dim=128)(edges)
    embeddings = embeddings.reshape(num_variables ** 2, -1)

    # Reshape the adjacency matrix
    adjacency = adjacency.reshape(num_variables ** 2, 1)

    # Apply common body
    num_layers = 5
    for i in range(3):
        embeddings = TransformerBlock(
            num_heads=4,
            key_size=64,
            embedding_size=128,
            init_scale=2. / num_layers,
            dropout_rate=0.,
            widening_factor=2,
            performer=False,
            linear=True,
            name=f'body_{i+1}'
        )(embeddings, adjacency, is_training)

    # Apply individual heads
    return GFlowNetOutput(
        logits=logits_head(embeddings, adjacency, is_training),
        stop=stop_head(embeddings, adjacency, is_training)
    )


def logits_head(embeddings, adjacency, is_training):
    num_layers = 5

    for i in range(2):
        embeddings = TransformerBlock(
            num_heads=4,
            key_size=64,
            embedding_size=128,
            init_scale=2. / num_layers,
            dropout_rate=0.,
            widening_factor=2,
            performer=False,
            linear=True,
            name=f'head_logits_{i+1}'
        )(embeddings, adjacency, is_training)

    logits = hk.nets.MLP([256, 128, 1])(embeddings)
    return jnp.squeeze(logits, axis=-1)


def stop_head(embeddings, adjacency, is_training):
    num_layers = 5

    for i in range(2):
        embeddings = TransformerBlock(
            num_heads=4,
            key_size=64,
            embedding_size=128,
            init_scale=2. / num_layers,
            dropout_rate=0.,
            widening_factor=2,
            performer=False,
            linear=True,
            name=f'head_stop_{i+1}'
        )(embeddings, adjacency, is_training)

    mean = jnp.mean(embeddings, axis=-2)  # Average over edges
    return hk.nets.MLP([256, 128, 1])(mean)


if __name__ == '__main__':
    from jax import random, vmap

    rng = hk.PRNGSequence(0)
    model = hk.transform(gflownet)

    adjacency = random.bernoulli(next(rng), 0.5, shape=(2, 3, 3))
    adjacency = adjacency.astype(jnp.float32)
    
    params = model.init(next(rng), adjacency[0], True)
    subkeys = jnp.array(rng.take(adjacency.shape[0]))
    outputs = vmap(model.apply, in_axes=(None, 0, 0, None))(params, subkeys, adjacency, True)
    print(outputs.logits.shape)
    print(outputs.stop.shape)

    @hk.transform
    def model(inputs):
        return gflownet(inputs, True)

    print(hk.experimental.tabulate(model)(adjacency[0]))
