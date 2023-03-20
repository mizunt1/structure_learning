import jax.numpy as jnp
from jax import jit, vmap, random
from jax.scipy.stats import multivariate_normal as jax_multivariate_normal
from jax.scipy.stats import norm as jax_normal

from mcmc_bs.mcmc_joint.utils import sel

class LinearGaussianGaussianJAX:
    """	
    LinearGaussianGaussianas above but using JAX and adjacency matrix representation	
    jit() and vmap() not yet implemented for this score as it tricky with indexing	
    """

    def __init__(self, *, obs_noise, mean_edge, sig_edge, verbose=False):
        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge
        self.verbose = verbose

        # init
        self.init_jax_functions()

    def init_jax_functions(self):

        # these will always have the same input shapes
        def log_marginal_likelihood_given_g_j_(j, w, data):

            n_samples, n_vars = data.shape
            isj = jnp.arange(n_vars) == j
            ispa = w[:, j] == 1

            data_j = sel(data, isj).sum(1)
            data_pa = sel(data, ispa)

            # mean
            mean_theta_j = jnp.where(ispa, self.mean_edge, 0)
            mean_j = data_pa @ mean_theta_j

            # cov
            # Note: `cov_j` is a NxN cov matrix, which can be huge for large N
            cov_theta_j = self.sig_edge ** 2.0 * sel(jnp.eye(n_vars), ispa)

            cov_j = self.obs_noise * jnp.eye(n_samples) + \
                data_pa @ cov_theta_j @ data_pa.T

            return jax_multivariate_normal.logpdf(x=data_j, mean=mean_j, cov=cov_j)

        self.log_marginal_likelihood_given_g_j = jit(
            vmap(log_marginal_likelihood_given_g_j_, (0, None, None), 0))

    def get_theta_shape(self, *, n_vars):
        """PyTree of parameter shape"""
        return jnp.array((n_vars, n_vars))

    def init_parameters(self, *, key, n_vars, n_particles, batch_size=0):
        """Samples batch of random parameters given dimensions of graph, from p(theta | G) 
        Returns:
            theta : [n_particles, n_vars, n_vars]
        """
        key, subk = random.split(key)
        
        if batch_size == 0:
            theta = self.mean_edge + self.sig_edge * random.normal(subk, shape=(n_particles, *self.get_theta_shape(n_vars=n_vars)))
        else:
            theta = self.mean_edge + self.sig_edge * random.normal(subk, shape=(batch_size, n_particles, *self.get_theta_shape(n_vars=n_vars)))

        return theta

    def log_marginal_likelihood_given_g(self, *, w, data, interv_targets):
        """Computes log p(x | G) in closed form using conjugacy properties of Dirichlet-Categorical	
            data : [n_samples, n_vars]	
            w:     [n_vars, n_vars]	
        """
        n_samples, n_vars = data.shape

        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                interv_targets,
                0.0,
                self.log_marginal_likelihood_given_g_j(jnp.arange(n_vars), w, data)
            )
        )
        # prev code without interventions
        # return jnp.sum(self.log_marginal_likelihood_given_g_j(jnp.arange(n_vars), w, data))

    def log_prob_parameters(self, *, theta, w):
        """p(theta | g); Assumes N(mean_edge, sig_edge^2) distribution for any given edge 
        In the linear Gaussian model, g does not matter.
            theta:          [n_vars, n_vars]
            w:              [n_vars, n_vars]
            interv_targets: [n_vars, ]
        """
        return jnp.sum(w * jax_normal.logpdf(x=theta, loc=self.mean_edge, scale=self.sig_edge))

    def log_likelihood(self, *, data, theta, w, interv_targets):
        """Computes p(x | theta, G). Assumes N(mean_obs, obs_noise) distribution for any given observation
            data :          [n_observations, n_vars]
            theta:          [n_vars, n_vars]
            w:              [n_vars, n_vars]
            interv_targets: [n_vars, ]
        """
        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets[None, ...],
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=data, loc=data @ (w * theta), scale=jnp.sqrt(self.obs_noise))
            )
        )
        # prev code without interventions
        # return jnp.sum(jax_normal.logpdf(x=data, loc=data @ (w * theta), scale=jnp.sqrt(self.obs_noise)))


