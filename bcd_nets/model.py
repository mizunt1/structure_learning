import sys
sys.path.append('./bcd_nets')
import logging

from abc import abstractmethod
from tqdm import tqdm
from typing import cast, OrderedDict
from termcolor import colored

import jax
from jax import numpy as jnp
from jax import config, jit, vmap, value_and_grad, grad
from jax.tree_util import tree_map, tree_multimap
from jax.flatten_util import ravel_pytree

import haiku as hk
import optax

from functools import partial
from tensorflow_probability.substrates.jax.distributions import Normal, Horseshoe

# from bcd_nets.flows import get_flow_CIF
from doubly_stochastic import GumbelSinkhorn
from utils import ff2, num_params, tau_schedule, lower, diag, get_W, get_p_model


class CheckTypesFilter(logging.Filter):
    """
        Class to prevent the logging of 
        `WARNING:root:The use of `check_types` is deprecated and does not have any effect.`
        while running the program
    """
    def filter(self, record):
        return "check_types" not in record.getMessage()


class BCD:
    def __init__(self, num_samples_posterior, model_obs_noise, args):

        """
            obs_noise: float
                Refers to noise variance of exogenous noise variables in the SCM
                That is, it refers to \sigma^2 in \eps \sim \mathcal{N}(0, \sigma^2 I)
        """
        config.update("jax_enable_x64", True)

        logger = logging.getLogger("root")
        logger.addFilter(CheckTypesFilter())

        assert 'gpu' in str(jax.devices()).lower()
        assert isinstance(model_obs_noise, float)
        
        key = jax.random.PRNGKey(args.seed)
        self.rng_key = key
        self.do_ev_noise = args.do_ev_noise
        assert self.do_ev_noise is True # right now only supports equal noise variance since obs_noise and args.data_obs_noise is a float

        self.num_variables = args.num_variables
        self.degree = args.num_edges // self.num_variables
        self.l_dim = self.num_variables * (self.num_variables - 1) // 2
        self.num_steps = args.num_steps
        self.update_freq = args.update_freq
        self.lr = args.lr
        self.num_samples = args.num_samples_data
        self.num_samples_posterior = num_samples_posterior
        self.batch_size = args.batch_size
        self.num_flow_layers = args.num_flow_layers
        self.flow_threshold = args.flow_threshold
        self.init_flow_std = args.init_flow_std
        self.use_alternative_horseshoe_tau = args.use_alternative_horseshoe_tau
        self.p_model_hidden_size = args.p_model_hidden_size
        self.num_perm_layers = args.num_perm_layers
        self.factorized = args.factorized
        self.use_flow = args.use_flow
        self.log_stds_max = args.log_stds_max
        self.max_deviation = args.max_deviation
        self.num_bethe_iters = args.num_bethe_iters
        self.logit_constraint = args.logit_constraint
        self.subsample = args.subsample
        self.s_prior_std = args.s_prior_std
        self.fixed_tau = args.fixed_tau
        self.edge_weight_threshold = args.edge_weight_threshold

        if self.do_ev_noise:    self.noise_dim = 1
        else:                   self.noise_dim = self.num_variables

        self._set_horseshoe_tau()
        self._set_optimizers()

        (   self.p_model, 
            self.P_params, 
            self.L_params, 
            self.L_states, 
            self.P_opt_params, 
            self.L_opt_params, 
            key
                            ) = self._init_params(key)

        self._set_tau()
        self.ds = GumbelSinkhorn(self.num_variables, noise_type="gumbel", tol=self.max_deviation)

    def _set_horseshoe_tau(self):
        if self.use_alternative_horseshoe_tau:
            p_n_over_n = 2 * self.degree / (self.num_variables - 1)
            if p_n_over_n > 1:  p_n_over_n = 1
            self.horseshoe_tau = p_n_over_n * jnp.sqrt(jnp.log(1.0 / p_n_over_n))
        else:
            self.horseshoe_tau = (1 / jnp.sqrt(self.num_samples)) * (2 * self.degree / ((self.num_variables - 1) - 2 * self.degree))
        
        if self.horseshoe_tau < 0:  
            self.horseshoe_tau = 1 / (2 * self.num_variables)
        
        print(f"Horseshoe tau is {self.horseshoe_tau}")

    def _set_tau(self):
        if self.fixed_tau is not None:  self.tau = self.fixed_tau
        else:                           self.tau = tau_schedule(0)

    def _set_optimizers(self):
        self.P_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-self.lr)]
        self.L_layers = [optax.scale_by_belief(eps=1e-8), optax.scale(-self.lr)]
        self.opt_P = optax.chain(*self.P_layers)
        self.opt_L = optax.chain(*self.L_layers)

    def _init_params(self, rng_key):
        if self.use_flow:
            _, self.sample_flow, self.get_flow_arrays, self.get_density = get_flow_CIF(
                jax.random.PRNGKey(rng_key),
                self.l_dim + self.noise_dim,
                self.num_flow_layers,
                self.batch_size,
                threshold=self.flow_threshold,
                init_std=self.init_flow_std,
                pretrain=False,
            )
            L_params, L_states = self.get_flow_arrays()   
        else:
            L_params = jnp.concatenate((jnp.zeros(self.l_dim), jnp.zeros(self.noise_dim), jnp.zeros(self.l_dim + self.noise_dim) - 1,))
            L_states = jnp.zeros((1,))

        P_params, p_model = get_p_model(
            rng_key, 
            self.num_variables, 
            self.batch_size, 
            self.num_perm_layers, 
            hidden_size=self.p_model_hidden_size, 
            do_ev_noise=self.do_ev_noise
        )
        rng_key = jax.random.split(rng_key)

        if self.factorized:  self.P_params = jnp.zeros((self.num_variables, self.num_variables))
        P_opt_params = self.opt_P.init(P_params)
        L_opt_params = self.opt_L.init(L_params)

        num_L_params = num_params(L_params)
        num_P_params = num_params(P_params)
        total_params = num_L_params + num_P_params

        print(f"BCD Nets has {ff2(total_params)} parameters in total")
        print(f"P model: {ff2(num_P_params)} parameters") 
        print(f"L model: {ff2(num_L_params)} parameters")
        print()
        return p_model, P_params, L_params, L_states, P_opt_params, L_opt_params, rng_key
    
    @partial(jit, static_argnums=(0,))
    def sample_L(self, rng_key, L_params, L_state):
        if self.use_flow:
            L_state = cast(hk.State, L_state)
            L_params = cast(hk.State, L_params)
            full_l_batch, full_log_prob_l, out_L_states = self.sample_flow(
                L_params, 
                L_state, 
                rng_key, 
                self.batch_size
            )
            return full_l_batch, full_log_prob_l, out_L_states
        else:
            L_params = cast(jnp.ndarray, L_params)
            means, log_stds = L_params[: self.l_dim + self.noise_dim], L_params[self.l_dim + self.noise_dim :]
            if self.log_stds_max is not None:
                # Do a soft-clip here to stop instability
                log_stds = jnp.tanh(log_stds / self.log_stds_max) * self.log_stds_max
            l_distribution = Normal(loc=means, scale=jnp.exp(log_stds))
            full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(self.batch_size,))
            full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)
            return full_l_batch, full_log_prob_l, None

    @partial(jit, static_argnums=(0,))
    def get_P_logits(self, rng_key, P_params, L_samples):
        if self.factorized:
            # We ignore L when giving the P parameters
            assert type(P_params) is jnp.ndarray
            p_logits = jnp.tile(P_params.reshape((1, self.num_variables, self.num_variables)), (len(L_samples), 1, 1))
        else:
            P_params = cast(hk.Params, P_params)
            p_logits = self.p_model(P_params, rng_key, L_samples)  # type:ignore

        if self.logit_constraint is not None:
            # Want to map -inf to -logit_constraint, inf to +logit_constraint
            p_logits = jnp.tanh(p_logits / self.logit_constraint) * self.logit_constraint

        return p_logits.reshape((-1, self.num_variables, self.num_variables))

    def log_prob_x(self, rng_key, data, log_sigmas, P, L):
        """Calculates log P(X|Z) for latent Zs

        X|Z is Gaussian so easy to calculate

        Args:
            data: an (n x dim)-dimensional array of observations
            log_sigmas: A (dim)-dimension vector of log standard deviations
            P: A (dim x dim)-dimensional permutation matrix
            L: A (dim x dim)-dimensional strictly lower triangular matrix
        Returns:
            log_prob: Log probability of observing `data` given P, L
        """
        if self.subsample:
            num_full_xs = len(data)
            X_batch_size = 16
            adjustment_factor = num_full_xs / X_batch_size
            data = jax.random.shuffle(rng_key, data)[:X_batch_size]
        else:
            adjustment_factor = 1
        n, dim = data.shape
        W = (P @ L @ P.T).T
        precision = (
            (jnp.eye(dim) - W) @ (jnp.diag(jnp.exp(-2 * log_sigmas))) @ (jnp.eye(dim) - W).T
        )
        eye_minus_W_logdet = 0
        log_det_precision = -2 * jnp.sum(log_sigmas) + 2 * eye_minus_W_logdet

        def datapoint_exponent(x):
            return -0.5 * x.T @ precision @ x

        log_exponent = vmap(datapoint_exponent)(data)

        return adjustment_factor * (
            0.5 * n * (log_det_precision - dim * jnp.log(2 * jnp.pi))
            + jnp.sum(log_exponent)
        )

    @partial(jit, static_argnums=(0,))
    def elbo(self, rng_key, P_params, L_params, L_states, data):
        """Computes ELBO estimate from parameters.

        Computes ELBO(P_params, L_params), given by
        E_{e1}[E_{e2}[log p(x|L, P)] - D_KL(q_L(P), p(L)) -  log q_P(P)],
        where L = g_L(L_params, e2) and P = g_P(P_params, e1).
        The derivative of this corresponds to the pathwise gradient estimator

        Args:
            P_params: inputs to sampling path functions
            L_params: inputs parameterising function giving L|P distribution
            data: (n x dim)-dimension array of inputs
            rng_key: jax prngkey object
            log_sigma_W: (dim)-dimensional array of log standard deviations
            log_sigma_l: scalar prior log standard deviation on (Laplace) prior on l.
        Returns:
            ELBO: Estimate of the ELBO
        """

        l_prior = Horseshoe(scale=jnp.ones(self.l_dim + self.noise_dim) * self.horseshoe_tau)
        rng_key, rng_key_1 = jax.random.split(rng_key, 2)

        full_l_batch, full_log_prob_l, out_L_states = self.sample_L(rng_key, L_params, L_states)
        w_noise = full_l_batch[:, -self.noise_dim:]
        l_batch = full_l_batch[:, :-self.noise_dim]
        batched_noises = jnp.ones((self.batch_size, self.num_variables)) * w_noise.reshape((self.batch_size, self.noise_dim))
        batched_L_samples = vmap(lower, in_axes=(0, None))(l_batch, self.num_variables)
        batched_P_logits = self.get_P_logits(rng_key_1, P_params, full_l_batch)

        batched_P_samples = self.ds.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key)
        # if soft sampling, use the line below
        # batched_P_samples = self.ds.sample_soft_batched_logits(batched_P_logits, self.tau, rng_key)

        vmapped_likelihood = vmap(self.log_prob_x, in_axes=(None, None, 0, 0, 0))
        likelihoods = vmapped_likelihood(rng_key, data, batched_noises, batched_P_samples, batched_L_samples)
        l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch)[:, :self.l_dim], axis=1)
        s_prior_probs = jnp.sum(full_l_batch[:, self.l_dim:] ** 2 / (2 * self.s_prior_std ** 2), axis=-1)
            
        KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs
        vmapped_P_logprob = vmap(self.ds.logprob, in_axes=(0, 0, None))
        logprob_P = vmapped_P_logprob(batched_P_samples, batched_P_logits, self.num_bethe_iters)
        log_P_prior = -jnp.sum(jnp.log(jnp.arange(self.num_variables) + 1))
        final_term = likelihoods - KL_term_L - logprob_P + log_P_prior

        return jnp.mean(final_term), out_L_states

    @partial(jit, static_argnums=(0,))
    def gradient_step(self, rng_key, P_params, L_params, L_states, P_opt_state, L_opt_state, data):
        rng_key, rng_key_2 = jax.random.split(rng_key, 2)
        tau_scaling_factor = 1.0 / self.tau

        (_, L_states), grads = value_and_grad(self.elbo, argnums=(1, 2), has_aux=True)(
                                    rng_key,
                                    P_params, 
                                    L_params, 
                                    L_states, 
                                    data
                                )
        elbo_grad_P, elbo_grad_L = tree_map(lambda x: -tau_scaling_factor * x, grads)
        
        l2_elbo_grad_P = grad(lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p)))(P_params)
        elbo_grad_P = tree_multimap(lambda x, y: x + y, elbo_grad_P, l2_elbo_grad_P)

        P_updates, P_opt_state = self.opt_P.update(elbo_grad_P, P_opt_state, P_params)
        P_params = optax.apply_updates(P_params, P_updates)
        L_updates, L_opt_state = self.opt_L.update(elbo_grad_L, L_opt_state, L_params)
        L_params = optax.apply_updates(L_params, L_updates)
        return rng_key_2, P_params, L_params, L_states, P_opt_state, L_opt_state

    @abstractmethod
    def train(self, data):
        pass


class Model(BCD):
    def __init__(self, num_samples_posterior, model_obs_noise, args):
        super(Model, self).__init__(num_samples_posterior, model_obs_noise, args)
    
    def train(self, data, seed):
        key = jax.random.PRNGKey(seed)
        data = jnp.array(data)
        with tqdm(range(self.num_steps), dynamic_ncols=True) as pbar:
            for i in pbar:
                text = colored('Learning q(G, θ, Σ)', 'yellow')
                pbar.set_description(text)
                (   self.rng_key, 
                    self.P_params, 
                    self.L_params, 
                    self.L_states, 
                    self.P_opt_state, 
                    self.L_opt_state
                                        ) = self.gradient_step(
                                                            self.rng_key, 
                                                            self.P_params, 
                                                            self.L_params, 
                                                            self.L_states, 
                                                            self.P_opt_params, 
                                                            self.L_opt_params, 
                                                            data
                                                        )
                
                if jnp.any(jnp.isnan(ravel_pytree(self.L_params)[0])):   print("Got NaNs in L params")

                if i % self.update_freq == 0:
                    if self.fixed_tau is None:   self.tau = tau_schedule(i)
                    current_elbo, _ = self.elbo(self.rng_key, self.P_params, self.L_params, self.L_states, data)
                    
                    postfix_dict = OrderedDict(
                        ELBO=f"{current_elbo}"
                    )
                    pbar.set_postfix(postfix_dict)

    def sample(self, seed):
        rng_key = jax.random.PRNGKey(seed)
        rounds = int(((self.num_samples_posterior // self.batch_size) + int(self.num_samples_posterior % self.batch_size != 0)))
        posterior_graphs = None
        posterior_thetas = None
        posterior_Sigmas = None
        with tqdm(range(rounds), dynamic_ncols=True, mininterval=1) as pbar:
            for i in pbar:
                text = colored('Sampling from q(G, θ, Σ)', 'yellow')
                pbar.set_description(text)
                rng_key, rng_key_1 = jax.random.split(rng_key, 2)
                full_l_batch, full_log_prob_l, out_L_states = self.sample_L(rng_key, self.L_params, self.L_states)
                w_noise = full_l_batch[:, -self.noise_dim:]
                l_batch = full_l_batch[:, :-self.noise_dim]
                batched_noises = jnp.ones((self.batch_size, self.num_variables)) * w_noise.reshape((self.batch_size, self.noise_dim))
                batched_lower_samples = vmap(lower, in_axes=(0, None))(l_batch, self.num_variables)
                batched_P_logits = self.get_P_logits(rng_key_1, self.P_params, full_l_batch)
                batched_P_samples = self.ds.sample_hard_batched_logits(batched_P_logits, self.tau, rng_key)
                
                posterior_theta_samples = vmap(get_W, (0, 0), (0))(batched_P_samples, batched_lower_samples)
                posterior_Sigma_samples = vmap(diag, (0), (0))(jnp.exp(2 * batched_noises))

                if posterior_thetas is None:
                    posterior_thetas = posterior_theta_samples
                    posterior_Sigmas = posterior_Sigma_samples

                else:
                    posterior_thetas = jnp.concatenate((posterior_thetas, posterior_theta_samples), axis=0)
                    posterior_Sigmas = jnp.concatenate((posterior_Sigmas, posterior_Sigma_samples), axis=0)

        posterior_graphs = jnp.where(jnp.abs(posterior_thetas) >= self.edge_weight_threshold, 1, 0)
        return posterior_graphs[:self.num_samples_posterior], posterior_thetas[:self.num_samples_posterior], posterior_Sigmas[:self.num_samples_posterior]


