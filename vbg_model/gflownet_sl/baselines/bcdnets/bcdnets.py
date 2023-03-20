import numpy as onp
import jax.numpy as jnp
from typing import Tuple, Optional, cast, Union
import itertools
import warnings

from gflownet_sl.baselines.bcdnets.doubly_stochastic import GumbelSinkhorn
import jax.random as rnd
from jax import vmap, grad, jit, lax, pmap, value_and_grad
from jax.tree_util import Partial as partial
from gflownet_sl.baselines.bcdnets.utils import (
    lower,
    eval_W_non_ev,
    eval_W_ev,
    ff2,
    num_params,
    save_params,
    rk,
    get_double_tree_variance,
    un_pmap,
    auroc,
)
from jax.tree_util import tree_map, tree_multimap

from tensorflow_probability.substrates.jax.distributions import (
    Normal,
    Horseshoe,
)

from jax import config
import haiku as hk
from gflownet_sl.baselines.bcdnets.models import (
    get_model,
    get_model_arrays,
)
import time
from jax.flatten_util import ravel_pytree
import optax
from gflownet_sl.baselines.bcdnets.flows import get_flow_CIF
from argparse import ArgumentParser
from gflownet_sl.baselines.bcdnets.metrics import intervention_distance, ensemble_intervention_distance
import jax

warnings.simplefilter(action="ignore", category=FutureWarning)
LStateType = Optional[hk.State]
PParamType = Union[hk.Params, jnp.ndarray]

print("finished imports")

config.update("jax_enable_x64", True)


PRNGKey = jnp.ndarray
QParams = Tuple[jnp.ndarray, hk.Params]


class BCDNets:
    def __init__(self,
                 Xs,
                 test_Xs=None,
                 ground_truth_W=None,
                 seed=0,
                 eval_eid = False,
                 do_ev_noise = True,
                 factorized = False,
                 use_flow = False,
                 batch_size = 64,
                 degree = 1,
                 subsample = False,
                 n_data = 100,
                 num_steps = 20_000,
                 lr = 1e-3,
                 logit_constraint = 10.0,
                 fixed_tau = 0.2,
                 max_deviation = .01,
                 use_alternative_horseshoe_tau = False,
                 sem_type = "linear-gauss",
                 posterior_sample_size=64):
        self.random_seed = seed
        self.eval_eid = eval_eid
        self.dim: int = Xs.shape[1]
        self.do_ev_noise = do_ev_noise
        self.factorized = factorized
        self.batch_size = batch_size
        self.degree = degree
        self.use_flow = use_flow
        self.subsample = subsample
        self.n_data = n_data
        assert sem_type in ["linear-gauss", "linear-gumbel"]
        self.sem_type = sem_type
        self.num_steps = num_steps
        self.lr = lr
        self.logit_constraint = logit_constraint
        self.max_deviation = max_deviation
        self.use_alternative_horseshoe_tau = use_alternative_horseshoe_tau

        override_to_cpu = False
        if override_to_cpu:
            jax.config.update("jax_platform_name", "cpu")

        onp.random.seed(self.random_seed)

        num_devices = jax.device_count()
        print(f"Number of devices: {num_devices}")
        if "gpu" not in str(jax.devices()).lower():
            print("NO GPU FOUND")
            # exit

        self.l_dim = self.dim * (self.dim - 1) // 2
        self.lr_P = lr
        self.lr_L = lr
        num_flow_layers = 2
        num_perm_layers = 2
        hidden_size = 128
        self.fixed_tau = fixed_tau

        num_mixture_components = 4
        num_outer = 1
        fix_L_params = False
        log_stds_max: Optional[float] = 10.0
        L_dist = Normal
        log_sigma_l = 0
        if do_ev_noise:
            # Generate noises same as GOLEM/Notears github
            log_sigma_W = jnp.zeros(self.dim)
        else:
            log_sigma_W = onp.random.uniform(low=0, high=jnp.log(2), size=(self.dim,))

        init_std = 0.00
        use_grad_global_norm_clipping = False
        P_norm = 100
        L_norm = 100

        flow_threshold = -1e3


        if do_ev_noise:
            noise_dim = 1
        else:
            noise_dim = self.dim

        init_flow_std = 0.1
        s_prior_std = 3.0
        calc_shd_c = False
        pretrain_flow = False
        rng_key = rk(self.random_seed)

        ds = GumbelSinkhorn(self.dim, noise_type="gumbel", tol=max_deviation)

        # This may be preferred from 'The horseshoe estimator: Posterior concentration around nearly black vectors'
        # van der Pas et al
        if use_alternative_horseshoe_tau:
            p_n_over_n = 2 * degree / (self.dim - 1)
            if p_n_over_n > 1:
                p_n_over_n = 1
            horseshoe_tau = p_n_over_n * jnp.sqrt(jnp.log(1.0 / p_n_over_n))
        else:
            horseshoe_tau = (1 / onp.sqrt(n_data)) * (2 * degree / ((self.dim - 1) - 2 * degree))
        if horseshoe_tau < 0:  # can happen for very small graphs
            horseshoe_tau = 1 / (2 * self.dim)
        print(f"Horseshoe tau is {horseshoe_tau}")

        self.ground_truth_W = ground_truth_W  # TODO input: (nvars*n_vars)
        # it has continuous values due to some of the evaluations, though we'll probably only need binary edges

        self.Xs = Xs  # TODO input: np.ndarray (n_samples*n_variables)
        self.test_Xs = test_Xs  # TODO optional input
        ground_truth_sigmas = jnp.exp(log_sigma_W)

        L_layers = []
        P_layers = []
        if use_grad_global_norm_clipping:
            L_layers += [optax.clip_by_global_norm(L_norm)]
            P_layers += [optax.clip_by_global_norm(P_norm)]
        P_layers += [optax.scale_by_belief(eps=1e-8), optax.scale(-self.lr_P)]
        L_layers += [optax.scale_by_belief(eps=1e-8), optax.scale(-self.lr_L)]
        opt_P = optax.chain(*P_layers)
        opt_L = optax.chain(*L_layers)
        opt_joint = None

        def init_parallel_params(rng_key: PRNGKey):
            @pmap
            def init_params(rng_key: PRNGKey):
                if use_flow:
                    L_params, L_states = get_flow_arrays()
                else:
                    L_params = jnp.concatenate(
                        (
                            jnp.zeros(self.l_dim),
                            jnp.zeros(noise_dim),
                            jnp.zeros(self.l_dim + noise_dim) - 1,
                        )
                    )
                    # Would be nice to put none here, but need to pmap well
                    L_states = jnp.array([0.0])
                P_params = get_model_arrays(
                    self.dim,
                    batch_size,
                    num_perm_layers,
                    rng_key,
                    hidden_size=hidden_size,
                    do_ev_noise=do_ev_noise,
                )
                if factorized:
                    P_params = jnp.zeros((self.dim, self.dim))
                P_opt_params = opt_P.init(P_params)
                L_opt_params = opt_L.init(L_params)
                return (
                    P_params,
                    L_params,
                    L_states,
                    P_opt_params,
                    L_opt_params,
                )

            rng_keys = jnp.tile(rng_key[None, :], (num_devices, 1))
            output = init_params(rng_keys)
            return output

        if use_flow:
            _, sample_flow, get_flow_arrays, get_density = get_flow_CIF(
                rk(0),
                self.l_dim + noise_dim,
                num_flow_layers,
                batch_size,
                num_mixture_components,
                threshold=flow_threshold,
                init_std=init_flow_std,
                pretrain=pretrain_flow,
                noise_dim=noise_dim,
            )

        _, p_model = get_model(
            self.dim, batch_size, num_perm_layers, hidden_size=hidden_size, do_ev_noise=do_ev_noise,
        )

        P_params, L_params, L_states, P_opt_params, L_opt_params = init_parallel_params(rng_key)
        rng_key = rnd.split(rng_key, num_devices)

        print(f"L model has {ff2(num_params(L_params))} parameters")
        print(f"P model has {ff2(num_params(P_params))} parameters")


        def get_P_logits(
            P_params: PParamType, L_samples: jnp.ndarray, rng_key: PRNGKey
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:

            if factorized:
                # We ignore L when giving the P parameters
                assert type(P_params) is jnp.ndarray
                p_logits = jnp.tile(P_params.reshape((1, self.dim, self.dim)), (len(L_samples), 1, 1))
            else:
                P_params = cast(hk.Params, P_params)
                p_logits = p_model(P_params, rng_key, L_samples)  # type:ignore

            if logit_constraint is not None:
                # Want to map -inf to -logit_constraint, inf to +logit_constraint
                p_logits = jnp.tanh(p_logits / logit_constraint) * logit_constraint

            return p_logits.reshape((-1, self.dim, self.dim))


        def sample_L(
            L_params: PParamType, L_state: LStateType, rng_key: PRNGKey,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, LStateType]:
            if use_flow:
                L_state = cast(hk.State, L_state)
                L_params = cast(hk.State, L_params)
                full_l_batch, full_log_prob_l, out_L_states = sample_flow(
                    L_params, L_state, rng_key, batch_size
                )
                return full_l_batch, full_log_prob_l, out_L_states
            else:
                L_params = cast(jnp.ndarray, L_params)
                means, log_stds = L_params[: self.l_dim + noise_dim], L_params[self.l_dim + noise_dim :]
                if log_stds_max is not None:
                    # Do a soft-clip here to stop instability
                    log_stds = jnp.tanh(log_stds / log_stds_max) * log_stds_max
                l_distribution = L_dist(loc=means, scale=jnp.exp(log_stds))
                if L_dist is Normal:
                    full_l_batch = l_distribution.sample(
                        seed=rng_key, sample_shape=(posterior_sample_size,)
                    )
                    full_l_batch = cast(jnp.ndarray, full_l_batch)
                else:
                    full_l_batch = (
                        rnd.laplace(rng_key, shape=(posterior_sample_size, self.l_dim + noise_dim))
                        * jnp.exp(log_stds)[None, :]
                        + means[None, :]
                    )
                full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)
                full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)

                out_L_states = None
                return full_l_batch, full_log_prob_l, out_L_states


        def log_prob_x(Xs, log_sigmas, P, L, rng_key):
            """Calculates log P(X|Z) for latent Zs

            X|Z is Gaussian so easy to calculate

            Args:
                Xs: an (n x dim)-dimensional array of observations
                log_sigmas: A (dim)-dimension vector of log standard deviations
                P: A (dim x dim)-dimensional permutation matrix
                L: A (dim x dim)-dimensional strictly lower triangular matrix
            Returns:
                log_prob: Log probability of observing Xs given P, L
            """
            if subsample:
                num_full_xs = len(Xs)
                X_batch_size = 16
                adjustment_factor = num_full_xs / X_batch_size
                Xs = rnd.shuffle(rng_key, Xs)[:X_batch_size]
            else:
                adjustment_factor = 1
            n, dim = Xs.shape
            W = (P @ L @ P.T).T
            precision = (
                (jnp.eye(dim) - W) @ (jnp.diag(jnp.exp(-2 * log_sigmas))) @ (jnp.eye(dim) - W).T
            )
            eye_minus_W_logdet = 0
            log_det_precision = -2 * jnp.sum(log_sigmas) + 2 * eye_minus_W_logdet

            def datapoint_exponent(x):
                return -0.5 * x.T @ precision @ x

            log_exponent = vmap(datapoint_exponent)(Xs)

            return adjustment_factor * (
                0.5 * n * (log_det_precision - dim * jnp.log(2 * jnp.pi))
                + jnp.sum(log_exponent)
            )


        def elbo(
            P_params: PParamType,
            L_params: hk.Params,
            L_states: LStateType,
            Xs: jnp.ndarray,
            rng_key: PRNGKey,
            tau: float,
            num_outer: int = 1,
            hard: bool = False,
        ) -> Tuple[jnp.ndarray, LStateType]:
            """Computes ELBO estimate from parameters.

            Computes ELBO(P_params, L_params), given by
            E_{e1}[E_{e2}[log p(x|L, P)] - D_KL(q_L(P), p(L)) -  log q_P(P)],
            where L = g_L(L_params, e2) and P = g_P(P_params, e1).
            The derivative of this corresponds to the pathwise gradient estimator

            Args:
                P_params: inputs to sampling path functions
                L_params: inputs parameterising function giving L|P distribution
                Xs: (n x dim)-dimension array of inputs
                rng_key: jax prngkey object
                log_sigma_W: (dim)-dimensional array of log standard deviations
                log_sigma_l: scalar prior log standard deviation on (Laplace) prior on l.
            Returns:
                ELBO: Estimate of the ELBO
            """
            num_bethe_iters = 20
            l_prior = Horseshoe(scale=jnp.ones(self.l_dim + noise_dim) * horseshoe_tau)
            # else:
            #     l_prior = Laplace(
            #         loc=jnp.zeros(l_dim + noise_dim),
            #         scale=jnp.ones(l_dim + noise_dim) * jnp.exp(log_sigma_l),
            #     )

            def outer_loop(rng_key: PRNGKey):
                """Computes a term of the outer expectation, averaging over batch size"""
                rng_key, rng_key_1 = rnd.split(rng_key, 2)
                full_l_batch, full_log_prob_l, out_L_states = sample_L(
                    L_params, L_states, rng_key
                )
                w_noise = full_l_batch[:, -noise_dim:]
                l_batch = full_l_batch[:, :-noise_dim]
                batched_noises = jnp.ones((batch_size, self.dim)) * w_noise.reshape(
                    (batch_size, noise_dim)
                )
                batched_lower_samples = vmap(lower, in_axes=(0, None))(l_batch, self.dim)
                batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key_1)
                if hard:
                    batched_P_samples = ds.sample_hard_batched_logits(
                        batched_P_logits, tau, rng_key,
                    )
                else:
                    batched_P_samples = ds.sample_soft_batched_logits(
                        batched_P_logits, tau, rng_key,
                    )
                likelihoods = vmap(log_prob_x, in_axes=(None, 0, 0, 0, None))(
                    Xs, batched_noises, batched_P_samples, batched_lower_samples, rng_key,
                )
                l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch)[:, :self.l_dim], axis=1)
                s_prior_probs = jnp.sum(
                    full_l_batch[:, self.l_dim:] ** 2 / (2 * s_prior_std ** 2), axis=-1
                )
                KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs
                logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(
                    batched_P_samples, batched_P_logits, num_bethe_iters
                )
                log_P_prior = -jnp.sum(jnp.log(onp.arange(self.dim) + 1))
                final_term = likelihoods - KL_term_L - logprob_P + log_P_prior

                return jnp.mean(final_term), out_L_states

            rng_keys = rnd.split(rng_key, num_outer)
            _, (elbos, out_L_states) = lax.scan(
                lambda _, rng_key: (None, outer_loop(rng_key)), None, rng_keys
            )
            elbo_estimate = jnp.mean(elbos)
            return elbo_estimate, tree_map(lambda x: x[-1], out_L_states)


        def eval_mean(
            P_params, L_params, L_states, Xs, rng_key=rk(0), do_shd_c=calc_shd_c, tau=1, get_dag_samples=False
        ):
            """Computes mean error statistics for P, L parameters and data"""
            P_params, L_params, L_states = (
                un_pmap(P_params),
                un_pmap(L_params),
                un_pmap(L_states),
            )

            if do_ev_noise:
                eval_W_fn = eval_W_ev
            else:
                eval_W_fn = eval_W_non_ev
            _, dim = Xs.shape
            x_prec = onp.linalg.inv(jnp.cov(Xs.T))
            full_l_batch, _, _ = sample_L(L_params, L_states, rng_key)
            w_noise = full_l_batch[:, -noise_dim:]
            l_batch = full_l_batch[:, :-noise_dim]
            batched_lower_samples = jit(vmap(lower, in_axes=(0, None)), static_argnums=(1,))(
                l_batch, dim
            )
            batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
            batched_P_samples = jit(ds.sample_hard_batched_logits)(
                batched_P_logits, tau, rng_key
            )

            def sample_W(L, P):
                return (P @ L @ P.T).T

            Ws = jit(vmap(sample_W))(batched_lower_samples, batched_P_samples)

            def sample_stats(W, noise):
                stats = eval_W_fn(
                    W,
                    ground_truth_W,
                    ground_truth_sigmas,
                    0.3,
                    Xs,
                    jnp.ones(dim) * jnp.exp(noise),
                    provided_x_prec=x_prec,
                    do_shd_c=do_shd_c,
                    do_sid=do_shd_c,
                )
                return stats

            stats = sample_stats(Ws[0], w_noise[0])
            stats = {key: [stats[key]] for key in stats}
            for i, W in enumerate(Ws[1:]):
                new_stats = sample_stats(W, w_noise[i])
                for key in new_stats:
                    stats[key] = stats[key] + [new_stats[key]]

            # stats = vmap(sample_stats)(rng_keys)
            out_stats = {key: onp.mean(stats[key]) for key in stats}
            out_stats["auroc"] = auroc(Ws, ground_truth_W, 0.3)

            return out_stats if not get_dag_samples else (out_stats, Ws)


        def get_num_sinkhorn_steps(P_params, L_params, L_states, rng_key):
            P_params, L_params, L_states, rng_key = (
                un_pmap(P_params),
                un_pmap(L_params),
                un_pmap(L_states),
                un_pmap(rng_key),
            )

            full_l_batch, _, _ = jit(sample_L)(L_params, L_states, rng_key)
            batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
            _, errors = jit(ds.sample_hard_batched_logits_debug)(
                batched_P_logits, tau, rng_key,
            )
            first_converged = jnp.where(jnp.sum(errors, axis=0) == -batch_size)[0]
            if len(first_converged) == 0:
                converged_idx = -1
            else:
                converged_idx = first_converged[0]
            return converged_idx


        def eval_ID(P_params, L_params, L_states, Xs, rng_key, tau):
            """Computes mean error statistics for P, L parameters and data"""
            P_params, L_params, L_states = (
                un_pmap(P_params),
                un_pmap(L_params),
                un_pmap(L_states),
            )

            _, dim = Xs.shape
            full_l_batch, _, _ = jit(sample_L, static_argnums=3)(L_params, L_states, rng_key)
            w_noise = full_l_batch[:, -noise_dim:]
            l_batch = full_l_batch[:, :-noise_dim]
            batched_lower_samples = jit(vmap(lower, in_axes=(0, None)), static_argnums=(1,))(
                l_batch, dim
            )
            batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
            batched_P_samples = jit(ds.sample_hard_batched_logits)(
                batched_P_logits, tau, rng_key,
            )

            def sample_W(L, P):
                return (P @ L @ P.T).T

            Ws = jit(vmap(sample_W))(batched_lower_samples, batched_P_samples)
            eid = ensemble_intervention_distance(
                ground_truth_W,
                Ws,
                onp.exp(log_sigma_W),
                onp.exp(w_noise) * onp.ones(dim),
                sem_type,
            )
            return eid


        @partial(
            pmap,
            axis_name="i",
            in_axes=(0, 0, 0, None, 0, None, None, None, None),
            static_broadcasted_argnums=(6, 7),
        )
        def parallel_elbo_estimate(P_params, L_params, L_states, Xs, rng_keys, tau, n, hard):
            elbos, _ = elbo(
                P_params, L_params, L_states, Xs, rng_keys, tau, n // num_devices, hard
            )
            mean_elbos = lax.pmean(elbos, axis_name="i")
            return jnp.mean(mean_elbos)


        @partial(
            pmap,
            axis_name="i",
            in_axes=(0, 0, 0, None, 0, 0, 0, None),
            static_broadcasted_argnums=(),
        )
        def parallel_gradient_step(
            P_params, L_params, L_states, Xs, P_opt_state, L_opt_state, rng_key, tau,
        ):
            rng_key, rng_key_2 = rnd.split(rng_key, 2)
            tau_scaling_factor = 1.0 / tau

            (_, L_states), grads = value_and_grad(elbo, argnums=(0, 1), has_aux=True)(
                P_params, L_params, L_states, Xs, rng_key, tau, num_outer, hard=True,
            )
            elbo_grad_P, elbo_grad_L = tree_map(lambda x: -tau_scaling_factor * x, grads)

            elbo_grad_P = lax.pmean(elbo_grad_P, axis_name="i")
            elbo_grad_L = lax.pmean(elbo_grad_L, axis_name="i")

            l2_elbo_grad_P = grad(
                lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p))
            )(P_params)
            elbo_grad_P = tree_multimap(lambda x, y: x + y, elbo_grad_P, l2_elbo_grad_P)

            P_updates, P_opt_state = opt_P.update(elbo_grad_P, P_opt_state, P_params)
            P_params = optax.apply_updates(P_params, P_updates)
            L_updates, L_opt_state = opt_L.update(elbo_grad_L, L_opt_state, L_params)
            if fix_L_params:
                pass
            else:
                L_params = optax.apply_updates(L_params, L_updates)

            return (
                P_params,
                L_params,
                L_states,
                P_opt_state,
                L_opt_state,
                rng_key_2,
            )


        @jit
        def compute_grad_variance(
            P_params, L_params, L_states, Xs, rng_key, tau,
        ):
            P_params, L_params, L_states, rng_key = (
                un_pmap(P_params),
                un_pmap(L_params),
                un_pmap(L_states),
                un_pmap(rng_key),
            )
            (_, L_states), grads = value_and_grad(elbo, argnums=(0, 1), has_aux=True)(
                P_params, L_params, L_states, Xs, rng_key, tau, num_outer, hard=True,
            )

            return get_double_tree_variance(*grads)


        def tau_schedule(i):
            boundaries = jnp.array([5_000, 10_000, 20_000, 60_000, 100_000])
            values = jnp.array([30.0, 10.0, 1.0, 1.0, 0.5, 0.25])
            index = jnp.sum(boundaries < i)
            return jnp.take(values, index)


        def get_histogram(L_params, L_states, P_params, rng_key):
            permutations = jax.nn.one_hot(
                jnp.vstack(list(itertools.permutations([0, 1, 2]))), num_classes=3
            )
            num_samples = 100
            if use_flow:
                full_l_batch, _, _ = jit(sample_flow, static_argnums=(3,))(
                    L_params, L_states, rng_key, 100
                )
                P_logits = get_P_logits(P_params, full_l_batch, rng_key)
            else:
                means, log_stds = (
                    L_params[: self.l_dim + noise_dim],
                    L_params[self.l_dim + noise_dim :],
                )
                l_distribution = L_dist(loc=means, scale=jnp.exp(log_stds))
                full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(num_samples,))
                assert type(full_l_batch) is jnp.ndarray
                P_logits = get_P_logits(P_params, full_l_batch, rng_key)
            batched_P_samples = jit(ds.sample_hard_batched_logits)(P_logits, tau, rng_key)
            histogram = onp.zeros(6)
            for P_sample in onp.array(batched_P_samples):
                for i, perm in enumerate(permutations):
                    if jnp.all(P_sample == perm):
                        histogram[i] += 1
            return histogram


        t0 = time.time()
        t_prev_batch = t0
        if fixed_tau is not None:
            tau = fixed_tau
        else:
            tau = tau_schedule(0)

        if use_flow:
            full_l_batch, log_prob_l, state = jit(sample_flow, static_argnums=(3,))(  # type: ignore
                un_pmap(L_params), un_pmap(L_states), rk(0), batch_size
            )


        soft_elbo = parallel_elbo_estimate(
            P_params, L_params, L_states, Xs, rng_key, tau, 100, False
        )[0]
        steps_t0 = time.time()
        best_elbo = -jnp.inf
        mean_dict = {}
        t00 = 0.0
        self.best_train_posterior = None

        for i in range(num_steps):
            (
                P_params,
                new_L_params,
                L_states,
                P_opt_params,
                new_L_opt_params,
                new_rng_key,
            ) = parallel_gradient_step(
                P_params, L_params, L_states, Xs, P_opt_params, L_opt_params, rng_key, tau,
            )
            if jnp.any(jnp.isnan(ravel_pytree(new_L_params)[0])):
                print("Got NaNs in L params")
            L_params = new_L_params
            L_opt_params = new_L_opt_params
            if i == 0:
                print(f"Compiled gradient step after {time.time() - t0}s")
                t00 = time.time()
            rng_key = new_rng_key

            if i % 400 == 0:
                if fixed_tau is None:
                    tau = tau_schedule(i)
                t000 = time.time()

                current_elbo = parallel_elbo_estimate(
                    P_params, L_params, L_states, Xs, rng_key, tau, 100, True,
                )[0]
                soft_elbo = parallel_elbo_estimate(
                    P_params, L_params, L_states, Xs, rng_key, tau, 100, False
                )[0]
                num_steps_to_converge = get_num_sinkhorn_steps(
                    P_params, L_params, L_states, rng_key
                )
                if i == 1:
                    print(f"Compiled estimates after {time.time() - t00}s")
                print(
                    f"After {i} iters, hard elbo is {ff2(current_elbo)}, soft elbo is {ff2(soft_elbo)}"
                )
                print(f"Iter time {ff2(time.time()-t_prev_batch)}s")
                print(
                    f"Took {ff2(time.time() - t000)}s to compute elbos, {num_steps_to_converge} sinkhorn steps"
                )
                out_dict = {
                    "ELBO": onp.array(current_elbo),
                    "soft ELBO": onp.array(soft_elbo),
                    "tau": onp.array(tau),
                    "Wall Time": onp.array(time.time() - t0),
                    "Sinkhorn steps": onp.array(num_steps_to_converge),
                }
                print(out_dict)
                t_prev_batch = time.time()

                if (i % 400 == 0) and (((time.time() - steps_t0) > 120) or (i % 4_000 == 0)):
                    # Log evalutation metrics at most once every two minutes
                    if (i % 10_000 == 0) and (i != 0):
                        _do_shd_c = False
                    else:
                        _do_shd_c = calc_shd_c
                    cache_str = f"_{sem_type.split('-')[1]}_d_{degree}_s_{self.random_seed}_{max_deviation}_{use_flow}.pkl"
                    # if time.time() - steps_t0 > 120:
                    #     # Don't cache too frequently
                    #     save_params(
                    #         P_params, L_params, L_states, P_opt_params, L_opt_params, cache_str,
                    #     )
                    #     print("cached_params")
                    elbo_grad_std = compute_grad_variance(
                        P_params, L_params, L_states, Xs, rng_key, tau,
                    )

                    try:
                        if test_Xs is not None:
                            mean_dict, dag_samples = eval_mean(
                                P_params, L_params, L_states, test_Xs, rk(i), _do_shd_c, get_dag_samples=True
                            )
                            train_mean_dict, train_dag_samples = eval_mean(
                                P_params, L_params, L_states, Xs, rk(i), _do_shd_c, get_dag_samples=True
                            )
                        else:
                            train_mean_dict, train_dag_samples = eval_mean(
                                P_params, L_params, L_states, Xs, rk(i), _do_shd_c, get_dag_samples=True
                            )
                            print("evaluating on trainset")
                            mean_dict = train_mean_dict
                            dag_samples = None
                    except:
                        print("Error occured in evaluating test statistics")
                        continue

                    if current_elbo > best_elbo:
                        best_elbo = current_elbo
                        best_shd = mean_dict["shd"]
                        print("best elbo: {:.2f} in iteration {:n}".format(onp.array(best_elbo), i))
                        print("best shd: {:.2f} in iteration {:n}".format(onp.array(best_shd), i))

                        train_dag_samples = onp.array(train_dag_samples) if train_dag_samples is not None else None
                        dag_samples = onp.array(dag_samples) if dag_samples is not None else None

                        save_params(
                            P_params, L_params, L_states, P_opt_params, L_opt_params, cache_str,
                            train_dag_samples, dag_samples, ground_truth_W
                        )
                        self.best_train_posterior = train_dag_samples
                        # DAG posterior dumped without thresholding
                        #
                        # threshold = .3
                        # dag_samples_clipped = np.where(np.abs(dag_samples) > threshold, dag_samples, 0)
                        # dag_samples, train_dag_samples


                    if eval_eid and i % 8_000 == 0:
                        t4 = time.time()
                        eid = eval_ID(P_params, L_params, L_states, Xs, rk(i), tau,)
                        print(f"EID_wass is {eid}, after {time.time() - t4}s")
                    print(f"MSE is {ff2(mean_dict['MSE'])}, SHD is {ff2(mean_dict['shd'])}")
                    metrics_ = (
                        {
                            "shd": mean_dict["shd"],
                            "shd_c": mean_dict["shd_c"],
                            "sid": mean_dict["sid"],
                            "mse": mean_dict["MSE"],
                            "tpr": mean_dict["tpr"],
                            "fdr": mean_dict["fdr"],
                            "fpr": mean_dict["fpr"],
                            "auroc": mean_dict["auroc"],
                            "ELBO Grad std": onp.array(elbo_grad_std),
                            "true KL": mean_dict["true_kl"],
                            "true Wasserstein": mean_dict["true_wasserstein"],
                            "sample KL": mean_dict["sample_kl"],
                            "sample Wasserstein": mean_dict["sample_wasserstein"],
                            "pred_size": mean_dict["pred_size"],
                            "train sample KL": train_mean_dict["sample_kl"],
                            "train sample Wasserstein": train_mean_dict["sample_wasserstein"],
                            "pred_size": mean_dict["pred_size"],
                        },
                    )

                    print(metrics_)
                    exit_condition = (
                        (i > 10_000)
                        and mean_dict["tpr"] < 0.5
                        and ((time.time() - steps_t0) > 3_600)
                    ) or ((mean_dict["shd"] > 300) and i > 10)
                    exit_condition = False
                    if exit_condition:
                        # While doing sweeps we don't want the runs to drag on for longer than
                        # would be reasonable to run them for
                        # So if runs are taking more than 20 mins per 400, we cut them
                        print(
                            f"Exiting after {time.time() - t0}s, avg time {(time.time() - t0) * 400 / (i + 1)}, tpr {mean_dict['tpr']}"
                        )
                        # sys.exit()

                    if use_flow:
                        full_l_batch, _, _ = jit(sample_flow, static_argnums=(3,))(  # type: ignore
                            un_pmap(L_params), un_pmap(L_states), rk(i), 10
                        )
                        P_logits = jit(get_P_logits)(
                            un_pmap(P_params), full_l_batch, un_pmap(rng_key)
                        )
                    else:
                        full_l_batch, _, _ = jit(sample_L, static_argnums=3)(
                            un_pmap(L_params), un_pmap(L_states), rk(i)
                        )
                        P_logits = jit(get_P_logits)(un_pmap(P_params), full_l_batch, rk(i))
                    batched_P_samples = jit(ds.sample_hard_batched_logits)(P_logits, tau, rk(i))
                    our_W = (
                        batched_P_samples[0]
                        @ lower(full_l_batch[0, :self.l_dim], self.dim)
                        @ batched_P_samples[0].T
                    ).T
                    batched_soft_P_samples = jit(ds.sample_soft_batched_logits)(
                        P_logits, tau, rk(i)
                    )
                    our_W_soft = (
                        batched_soft_P_samples[0]
                        @ lower(full_l_batch[0, :self.l_dim], self.dim)
                        @ batched_soft_P_samples[0].T
                    ).T

                    print(f"Max value of P_logits was {ff2(jnp.max(jnp.abs(P_logits)))}")
                    if self.dim == 3:
                        print(get_histogram(L_params, L_states, P_params, rng_key))
                    steps_t0 = time.time()


if __name__ == '__main__':
    from gflownet_sl.utils.graph import sample_erdos_renyi_linear_gaussian, get_weighted_adjacency
    from gflownet_sl.utils.sampling import sample_from_linear_gaussian
    from numpy.random import default_rng

    seed = 1
    rng = default_rng(seed)
    expected_degree = 2  # 1
    graph = sample_erdos_renyi_linear_gaussian(
        num_variables=32,
        num_edges=expected_degree*32/2,  # each extra nvars/2 adds 1 to avg node degree
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=1,  # 0.1,
        rng=rng
    )
    data = sample_from_linear_gaussian(
        graph,
        num_samples=100,
        rng=rng
    )

    w_adjacency = get_weighted_adjacency(graph)

    BCDNets(Xs=data.to_numpy(),
            test_Xs=data.to_numpy(),
            ground_truth_W=w_adjacency)
