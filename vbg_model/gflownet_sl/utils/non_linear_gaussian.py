import numpy as np
import jax.numpy as jnp
import haiku as hk
import jax
import networkx as nx

from collections import namedtuple
from scipy.stats import norm


@hk.without_apply_rng
@hk.transform
def model(inputs, parents):
    # Following the architecture from DiBS
    # https://arxiv.org/pdf/2105.11839.pdf (Section 6.3)
    outputs = hk.nets.MLP(
        (5, 1),
        activation=jax.nn.relu,
        with_bias=True,
        activate_final=False,
        name='mlp'
    )(inputs * parents)
    return jnp.squeeze(outputs, axis=-1)


NormalParameters = namedtuple('NormalParameters', ['loc', 'scale'])


class NonLinearGaussian:
    def __init__(self, model=model, obs_noise=0.1):
        self.model = model
        self.obs_noise = obs_noise

    def sample_ground_truth_parameters(self, key, adjacency):
        num_variables = adjacency.shape[0]

        # Sample (unmasked) parameters
        subkeys = jax.random.split(key, num_variables)
        inputs = jnp.zeros((num_variables, 1, num_variables))
        params = jax.vmap(self.model.init)(subkeys, inputs, adjacency.T)

        # Mask the weights of the first layer
        weights = params['mlp/~/linear_0']['w']
        weights = weights * jnp.expand_dims(adjacency.T, axis=2)

        return hk.data_structures.merge(params, {'mlp/~/linear_0': {'w': weights}})

    def sample_data(self, key, params, adjacency, num_samples):
        num_variables = adjacency.shape[0]
        samples = jnp.zeros((num_samples, num_variables))
        subkeys = jax.random.split(key, num_variables)

        # Ancestral sampling
        graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        for node, subkey in zip(nx.topological_sort(graph), subkeys):
            # Forward pass of the MLP to get the mean of the observations
            node_params = jax.tree_util.tree_map(lambda param: param[node], params)
            node_mean = self.model.apply(node_params, samples, adjacency[:, node])

            # Sample from a Normal distribution
            epsilon = jax.random.normal(subkey, shape=node_mean.shape)
            node_samples = node_mean + self.obs_noise * epsilon
            samples = samples.at[:, node].set(node_samples)
        
        return np.asarray(samples)

    def sample_thetas(self, key, params, num_samples):
        leaves, treedef = jax.tree_util.tree_flatten(params.loc)
        subkeys = jax.random.split(key, len(leaves))
        subkeys = jax.tree_util.tree_unflatten(treedef, subkeys)

        epsilons = jax.tree_util.tree_map(
            lambda param, subkey: jax.random.normal(subkey,
            shape=(num_samples,) + param.shape), params.loc, subkeys)
        return jax.tree_util.tree_map(
            lambda loc, scale, epsilon: loc + scale * epsilon,
            params.loc, params.scale, epsilons)

    def log_likelihood(self, adjacencies, thetas, data):
        # Compute log P(D | G, theta)
        v_model = jax.vmap(self.model.apply, in_axes=(None, None, 0))  # vmapping over graphs
        v_model = jax.vmap(v_model, in_axes=(0, None, None))  # vmapping over thetas
        v_model = jax.vmap(v_model, in_axes=(0, None, 1), out_axes=1)  # vmapping over variables

        means = v_model(thetas, data, adjacencies)  # (num_graphs, num_thetas, num_samples, num_variables)
        data = jnp.broadcast_to(data, means.shape)
        log_likelihoods = norm.logpdf(data, loc=means, scale=self.obs_noise)
        return jnp.sum(log_likelihoods, axis=(2, 3))

    def kl_divergence(self, adjacencies, params):
        # Compute KL(q(theta | G) || P(theta | G)), where P(theta | G) = N(0, I)
        def _partition_first_layer_weights(tree):
            first_weights, other_params = hk.data_structures.partition(
                lambda module_name, name, _: (module_name == 'mlp/~/linear_0') and (name == 'w'),
                tree
            )
            return (first_weights['mlp/~/linear_0']['w'], other_params)

        def _kl(loc, scale):
            # From https://arxiv.org/abs/1312.6114 (Appendix B)
            return -0.5 * 1 + 2 * jnp.log(scale) - (loc ** 2) - (scale ** 2)

        def _kl_divergence(adjacency, first_weights, other_params):
            # Compute the KL-divergence for the weights of the first layer.
            # Mask the components, depending on the adjacency matrix
            kls_first = _kl(*first_weights)
            kls_first = jnp.sum(kls_first * jnp.expand_dims(adjacency.T, axis=-1))

            # Compute the KL-divergence for all other parameters
            kls_other = jax.tree_util.tree_map(_kl, *other_params)
            kls_other = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y), kls_other)

            return kls_first + kls_other

        # Partition the model parameters into the weights of the first layer, and the rest
        first_weights, other_params = map(lambda args: NormalParameters(*args),
            zip(*map(_partition_first_layer_weights, params)))

        v_kl_divergence = jax.vmap(_kl_divergence, in_axes=(0, None, None))  # vmapping over graphs
        v_kl_divergence = jax.vmap(v_kl_divergence, in_axes=(None, 0, 0))  # vmapping over variables

        kl_divergences = v_kl_divergence(adjacencies, first_weights, other_params)
        return jnp.sum(kl_divergences, axis=1)

    def loss(self, params, key, adjacencies, data, num_samples_thetas):
        # Sample parameters thetas for the MC estimate
        thetas = self.sample_thetas(key, params, num_samples_thetas)

        # Compute the log-likelihood and the KL-divergence
        log_likelihoods = self.log_likelihood(adjacencies, thetas, data)
        kl_divergences = self.kl_divergence(adjacencies, params)

        # The loss is the negative ELBO. TODO: Assume uniform P(G)
        expected_log_likelihood = jnp.mean(log_likelihoods, axis=1)  # Expectation over theta
        return -jnp.mean(expected_log_likelihood - kl_divergences)  # Expectation over graphs

    def init(self, num_variables, loc=0., scale=1.):
        # Sample dummy model parameters
        subkeys = jnp.zeros((num_variables, 2), dtype=jnp.uint32)  # Dummy random keys
        inputs = jnp.zeros((num_variables, 1, num_variables))
        adjacency = jnp.zeros((num_variables, num_variables))
        dummy = jax.vmap(self.model.init)(subkeys, inputs, adjacency)
        return NormalParameters(
            loc=jax.tree_util.tree_map(
                lambda param: jnp.full_like(param, loc), dummy),
            scale=jax.tree_util.tree_map(
                lambda param: jnp.full_like(param, scale), dummy),
        )
