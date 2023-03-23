import numpy as np
import haiku as hk
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

def get_weighted_adjacency(graph):
    adjacency = np.zeros((len(graph), len(graph)), dtype=np.float_)
    index = dict((node, idx) for (idx, node) in enumerate(graph.nodes))
    for v in graph.nodes:
        cpd = graph.get_cpds(v)
        for u, weight in zip(cpd.evidence, cpd.mean[1:]):
            adjacency[index[u], index[v]] = weight
    return adjacency

def edge_marginal_means(means, adjacency_matrices):
    num_variables = adjacency_matrices.shape[1]
    num_matrices =  adjacency_matrices.shape[0]
    adjacency_expectation = np.sum(adjacency_matrices, axis=0)/num_matrices
    edge_marginal_means = np.zeros((num_variables, num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            edge_marginal_means[i][j] = adjacency_expectation[i][j] * means[i][j].item()
    return edge_marginal_means


#  BCD nets
def npperm(M):
    # From user lesshaste on github: https://github.com/scipy/scipy/issues/7151
    n = M.shape[0]
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(n)
    v = M.sum(axis=0)
    p = np.prod(v)
    while j < n - 1:
        v -= 2 * d[j] * M[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s * prod
        f[0] = 0
        f[j] = f[j + 1]
        f[j + 1] = j + 1
        j = f[0]
    return p / 2 ** (n - 1)

def ff2(x):
    if type(x) is str:
        return x
    if np.abs(x) > 1000 or np.abs(x) < 0.1:
        return np.format_float_scientific(x, 3)
    else:
        return f"{x:.2f}"

def num_params(params: hk.Params) -> int:
    return len(ravel_pytree(params)[0])

def tau_schedule(i):
    boundaries = jnp.array([5_000, 10_000, 20_000, 60_000, 100_000])
    values = jnp.array([30.0, 10.0, 1.0, 1.0, 0.5, 0.25])
    index = jnp.sum(boundaries < i)
    return jnp.take(values, index)

def lower(theta, dim):
    """Given n(n-1)/2 parameters theta, form a
    strictly lower-triangular matrix"""
    out = jnp.zeros((dim, dim))
    out = out.at[jnp.triu_indices(dim, 1)].set(theta)
    out = out.T
    return out

@jax.jit
def diag(noises):
    return jnp.diag(noises)


@jax.jit
def get_W(P, L):
    return (P @ L @ P.T).T


def get_p_model(rng_key, dim, batch_size, num_layers, hidden_size=32, do_ev_noise=True):
    if do_ev_noise: noise_dim = 1
    else:           noise_dim = dim
    l_dim = dim * (dim - 1) // 2
    input_dim = l_dim + noise_dim

    def forward_fn(in_data: jnp.ndarray) -> jnp.ndarray:
        # Must have num_heads * key_size (=64) = embedding_size
        x = hk.Linear(hidden_size)(hk.Flatten()(in_data))
        x = jax.nn.gelu(x)
        for _ in range(num_layers - 2):
            x = hk.Linear(hidden_size)(x)
            x = jax.nn.gelu(x)
        x = hk.Linear(hidden_size)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(dim * dim)(x)

    forward_fn_init, forward_fn_apply = hk.transform(forward_fn)
    blank_data = np.zeros((batch_size, input_dim))
    laplace_params = forward_fn_init(rng_key, blank_data)
    return laplace_params, forward_fn_apply