import jax.numpy as jnp
from scipy.special import gammaln

import pickle
import optax
import numpy as np
import math
import jax 
from jax import nn, lax, scipy
from collections import namedtuple
import bisect

GFlowNetOutput = namedtuple('GFlowNetOutput', ['logits', 'stop'])
MASKED_VALUE = -1e5
NormalParameters = namedtuple('NormalParameters', ['mean', 'precision'])
# precision is 1/(sigma**2) 1/variance


def mask_logits(logits, masks):
    return masks * logits + (1. - masks) * MASKED_VALUE


def detailed_balance_loss(
        log_pi_t,
        log_pi_tp1,
        actions,
        rewards,
        num_edges,
        subsq,
        subsq_mask,
        delta=1.
    ):
    # Compute the forward log-probabilities
    log_pF = jnp.take_along_axis(log_pi_t, actions, axis=-1)

    # Compute the backward log-probabilities
    log_pB = -jnp.log1p(num_edges)

    # Total of "log R(s') + log P_B(s | s') - log R(s) - log P_F(s' | s)"
    # Recall that "rewards" here is "log R(s') - log R(s)"
    total = jnp.squeeze(rewards + log_pB - log_pF, axis=-1)
    # Sum the total above along all subsequences
    subsq_total = jnp.zeros_like(subsq_mask)
    subsq_total = subsq_total.at[subsq.rows].add(total[subsq.cols])
    
    # Compute the log-probabilities of terminating at the start for all subsequences
    subsq_log_psf_start = log_pi_t[subsq.start, :, -1]

    # Compute the log-probabilities of terminating at the end of all subsquences
    subsq_log_psf_end = log_pi_tp1[subsq.end, :, -1]

    error = subsq_total + subsq_log_psf_start - lax.stop_gradient(subsq_log_psf_end)
    # Weighted mean, weighted by the mask over subsequences
    losses = optax.huber_loss(error, delta=delta)
    loss = jnp.sum(losses * subsq_mask) / jnp.sum(subsq_mask)

    logs = {
        'error': error,
        'loss': loss,
        'mask': subsq_mask
    }

    return (loss, logs)


def log_policy(outputs, masks):
    masks = masks.reshape(outputs.logits.shape)
    masked_logits = mask_logits(outputs.logits, masks)
    can_continue = jnp.any(masks, axis=-1, keepdims=True)

    logp_continue = (nn.log_sigmoid(-outputs.stop)
        + nn.log_softmax(masked_logits, axis=-1))
    logp_stop = nn.log_sigmoid(outputs.stop)

    # In case there is no valid action other than stop
    logp_continue = jnp.where(can_continue, logp_continue, MASKED_VALUE)
    logp_stop = logp_stop * can_continue

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)


def uniform_log_policy(masks):
    masks = masks.reshape(masks.shape[0], -1)
    num_edges = jnp.sum(masks, axis=-1, keepdims=True)

    logp_stop = -jnp.log1p(num_edges)
    logp_continue = mask_logits(logp_stop, masks)

    return jnp.concatenate((logp_continue, logp_stop), axis=-1)

def compute_delta_score_lingauss(adjacency, action, params, prior, XTX, obs_noise, use_prior=True):
    num_variables = params.mean.shape[0]
    source, target = divmod(action, num_variables)
    term1 = -2 * params.mean[source, target] * XTX[source, target]
    is_not_zero = params.precision == 0  
    var = (1/params.precision)*is_not_zero 
    moment_2 = var[source, target] + params.mean[source, target] ** 2
    term2 = XTX[source, source] * moment_2
    term3 = 2 * params.mean[source, target] * jnp.vdot(
        XTX[source], adjacency[:, target] * params.mean[:, target])
    if prior:
        prior_mean = prior.mean[source, target]
        prior_precision = prior.precision[source, target]
        term4 = prior_precision * (params.mean[source, target] - prior_mean) ** 2
        
        term5 = (jnp.log(2 * jnp.pi) - jnp.log(prior_precision)
                 + prior_precision * var[source, target])
        return -0.5 * ((term1 + term2 + term3) / (obs_noise ** 2) + term4 + term5)
    else:
        return -0.5 * (term1 + term2 + term3) / (obs_noise ** 2)

    

def compute_delta_score_lingauss_full(adjacency, action, params,
                                      prior, XTX, obs_noise,
                                      weight, use_erdos_prior):
    num_variables = params.mean.shape[0]
    source, target = divmod(action, num_variables)
    adjacency = adjacency.at[source, target].set(1)
    precision = params.precision[:,:,target][:,:,0]
    # masking covariance terms for R(G)
    mask_cov = jnp.zeros((num_variables, num_variables))
    mask_cov = mask_cov.at[source, :].set(1)
    mask_cov = mask_cov.at[:, source].set(1)
    precision_masked = precision - precision*mask_cov
    precision_masked = precision_masked.at[source, source].set(1)
    g_dash_cov = jnp.linalg.inv(precision)
    g_cov = jnp.linalg.inv(precision_masked)
    # difference of likelihood terms, term1, term2, term3
    term1 = -2 * params.mean[source, target] * XTX[source, target]
    moment_2 = g_dash_cov[source, source] + params.mean[source, target] ** 2
    term2 = XTX[source, source] * moment_2
    moment_3 = adjacency[:, target] * (
        params.mean[source, target] * params.mean[:, target] + g_dash_cov[source].T)
    # subtracting k is not i term from dot product 

    offset = 2*(params.mean[source, target]**2 + g_dash_cov[source, source])*XTX[source, source]
    term3 = 2 * jnp.vdot(XTX[source], moment_3.squeeze(1)) - offset
    g_dash_mean =  params.mean[:,target].squeeze(1) - prior.mean 
    # Masking mean term for R(G)
    mask = jnp.zeros((num_variables, num_variables))
    mask = mask.at[source,target].set(1)
    g_mean = (params.mean[:,target]).squeeze(1) - ((params.mean*mask)[:,target]).squeeze(1) - prior.mean
    inv_prior_precision = jnp.linalg.inv(prior.precision)
    #KL terms
    kl_1 = jnp.matmul(g_dash_mean, jnp.matmul(prior.precision, g_dash_mean)) - jnp.matmul(g_mean, jnp.matmul(prior.precision, g_mean))
    kl_2 = jnp.trace(jnp.matmul(prior.precision, g_dash_cov)) - jnp.trace(jnp.matmul(prior.precision, g_cov))
    kl_3 = -1*((jnp.linalg.slogdet(g_dash_cov))[1] - (jnp.linalg.slogdet(g_cov)[1]))
    kl_term = 0.5*(kl_1 + kl_2 + kl_3)
    # prior term
    # Key before adding the new source node
    parents_before = (jnp.sum(adjacency[:,target])).astype(int)

    parents_after = parents_before + 1
    # Key after adding the new source node
    if use_erdos_prior:
        prior_score_before = erdos_renyi_prior(num_variables)[parents_before]
        prior_score_after = erdos_renyi_prior(num_variables)[parents_after]
        return  weight * (-0.5 * ((term1 + term2 + term3) / (obs_noise)) - kl_term + prior_score_after - prior_score_before)
    else:
        return  weight * (-0.5 * (weight*(term1 + term2 + term3) / (obs_noise))  - kl_term)



def erdos_renyi_prior(num_variables):
    num_edges_per_node = 1
    num_edges = num_variables * num_edges_per_node  # Default value
    p = num_edges / ((num_variables * (num_variables - 1)) // 2)
    all_parents = jnp.arange(num_variables)
    log_prior = (all_parents * math.log(p) + (num_variables - all_parents - 1) * math.log1p(-p))
    return log_prior

def fair_prior(num_variables):
    all_parents = jnp.arange(5)
    log_prior = (
        - scipy.special.gammaln(num_variables + 1)
        + scipy.special.gammaln(num_variables - all_parents + 1)
        + scipy.special.gammaln(all_parents + 1)
    )
    return log_prior

def edge_marginal_means(means, adjacency_matrices):
    num_variables = adjacency_matrices.shape[1]
    num_matrices =  adjacency_matrices.shape[0]
    adjacency_expectation = np.sum(adjacency_matrices, axis=0)/num_matrices
    edge_marginal_means = np.zeros((num_variables, num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            edge_marginal_means[i][j] = adjacency_expectation[i][j] * means[i][j].item()
    return edge_marginal_means


def update_parameters(params, prior, graphs, empirical_cov, obs_noise):
    # Compute the marginals for the posterior approximation over graphs
    w = jnp.mean(graphs, axis=0)
    m = jnp.einsum('nij,nkj->ikj', graphs, graphs) / graphs.shape[0]

    # Update the variance
    diag_cov = jnp.expand_dims(jnp.diag(empirical_cov), axis=1)
    inv_variance = w*prior.precision + (w * diag_cov / (obs_noise**2))
    # Update the mean
    is_zero = inv_variance == 0
    inv_variance = inv_variance*is_zero + is_zero + inv_variance

    term1 = jnp.einsum('ikj,kj,kj->ij', m, empirical_cov, params.mean)
    term2 = w * empirical_cov * (1 + params.mean)
    mean = (prior.mean *prior.precision *w  + (term2 - term1) / obs_noise**2) / inv_variance   
    return NormalParameters(mean=mean, precision=inv_variance)

def update_parameters_full(prior, graphs, X, obs_noise):
    num_graphs = graphs.shape[0]
    XTX = jnp.matmul(X.T, X)
    def _update(parents, y):
        # This is equivalent to Bayesian Linear Regression, but where the
        # inputs are weighted by the edge marginals.
        XTX_w = jnp.matmul(parents.T, parents) * XTX / num_graphs
        XTy_w = jnp.mean(parents, axis=0) * jnp.dot(X.T, y)
        precision = prior.precision + XTX_w / (obs_noise)
        b = jnp.dot(prior.precision, prior.mean) + XTy_w / (obs_noise)
        mean = jnp.linalg.solve(precision, b)
        return NormalParameters(mean=mean, precision=precision)
    return jax.vmap(_update, in_axes=-1, out_axes=-1)(graphs, X)
