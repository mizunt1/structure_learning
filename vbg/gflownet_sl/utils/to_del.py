def compute_delta_score_lingauss(adjacency, action, params, XTX, obs_noise):
    num_variables = params.mean.shape[0]
    source, target = divmod(action, num_variables)
    term1 = -2 * params.mean[source, target] * XTX[source, target]
    moment_2 = 1. / params.precision[source, target] + params.mean[source, target] ** 2
    term2 = XTX[source, source] * moment_2
    term3 = 2 * params.mean[source, target] * jnp.vdot(
        XTX[source], adjacency[:, target] * params.mean[:, target])
 
    return -0.5 * (term1 + term2 + term3) / (obs_noise ** 2)

def update_parameters(params, prior, graphs, empirical_cov, obs_noise):
    # Compute the marginals for the posterior approximation over graphs                                                                                                                                             
    w = jnp.mean(graphs, axis=0)
    m = jnp.einsum('nij,nkj->ikj', graphs, graphs) / graphs.shape[0]

    # Update the variance                                                                                                                                                                                           
    diag_cov = jnp.expand_dims(jnp.diag(empirical_cov), axis=1)
    inv_variance = prior.precision + w * diag_cov / obs_noise

    # Update the mean                                                                                                                                                                                               
    term1 = jnp.einsum('ikj,kj,kj->ij', m, empirical_cov, params.mean)
    term2 = w * empirical_cov * (1 + params.mean)
    mean = (prior.mean *prior.precision + (term2 - term1) / obs_noise) / inv_variance
    return NormalParameters(mean=mean, precision=inv_variance)

