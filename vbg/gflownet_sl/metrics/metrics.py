import numpy as np
import jax.numpy as jnp
from scipy.special import logsumexp

# from dibs.metrics import ParticleDistribution

from sklearn import metrics

from scipy.stats import bootstrap, t


# def get_empirical(g):
#     # Fixes a bug in MarginalDiBS.get_empirical
#     unique, counts = np.unique(g, axis=0, return_counts=True)
#     logp = jnp.log(counts) - jnp.log(g.shape[0])
#     return ParticleDistribution(logp=logp, g=unique)


def get_mean_and_ci(mean, bs_results, num_samples, alpha=0.95):
    t_ = t.ppf(1 - (1 - alpha) / 2, df=num_samples - 1)
    return (mean, mean)


def pairwise_structural_hamming_distance(*, x, y):
    """
    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
    This means, edge reversals do not double count, and that getting an undirected edge wrong only counts 1

    Args:
        x (ndarray): batch of adjacency matrices  [N, d, d]
        y (ndarray): batch of adjacency matrices  [M, d, d]

    Returns:
        matrix of shape ``[N, M]``  where elt ``i,j`` is  SHD(``x[i]``, ``y[j]``)
    """

    # all but first axis is usually used for the norm, assuming that first dim is batch dim
    
    assert(x.ndim == 3 and y.ndim == 3)

    # via computing pairwise differences
    pw_diff = jnp.abs(jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0))
    pw_diff = pw_diff + pw_diff.transpose((0, 1, 3, 2))

    # ignore double edges
    pw_diff = jnp.where(pw_diff > 1, 1, pw_diff)
    shd = jnp.sum(pw_diff, axis=(2, 3)) / 2

    return shd

def calibration(graphs, ground_truth_graph, bin_size=0.1):
    """
    This function gives the calibration between n sampled graphs
    and the ground truth graph, returning the ECE score,
    which corresponds at the difference between the
    probability of edges in predicted graph from samples of the 
    posterior and the ground truth graph.
    As was done in https://arxiv.org/pdf/2205.12934.pdf

    Parameters
    ----------
    graphs: np.array(n, d, d)
        samples of predicted graphs from posterior as 0/1 adjacency
        matrices.  n is number of samples, d is  number of edges
    ground_truth_graph: np.array(d, d)
        ground truth graph.
    bin_size: float
        size of a single bin. between 0 and 1.

    Returns 
    -------
    score: float
        ECE score
    """
    score = 0
    edge_marginals = np.sum(graphs, axis=0)/(graphs.shape[0])
    bins = np.arange(0, 1, bin_size)
    for bin_ in bins:
        if bin_ + bin_size == 1.0:
            masked_graph = edge_marginals * ((edge_marginals >= bin_)  & (edge_marginals <= 1.0))
        else:
            masked_graph = edge_marginals * ((edge_marginals >= bin_) & (edge_marginals < (bin_ + bin_size)))
        non_zero = masked_graph * (masked_graph != 0)
        ground_truth_sum = np.sum(ground_truth_graph * non_zero)
        predicted_sum = np.sum(masked_graph)
        difference = np.abs(ground_truth_sum - predicted_sum)
        score += difference
    return score / ground_truth_graph.shape[0]**2

def expected_shd(posterior, ground_truth):
    """Compute the Expected Structural Hamming Distance.
    This function computes the Expected SHD between a posterior approximation
    given as a collection of samples from the posterior, and the ground-truth
    graph used in the original data generation process.
    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.
    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.
    Returns
    -------
    e_shd : float
        The Expected SHD.
    """
    # Compute the pairwise differences
    diff = np.abs(posterior - np.expand_dims(ground_truth, axis=0))
    diff = diff + diff.transpose((0, 2, 1))

    # Ignore double edges
    diff = np.minimum(diff, 1)
    shds = np.sum(diff, axis=(1, 2)) / 2

    return np.mean(shds)


def expected_edges(posterior):
    """Compute the expected number of edges.
    This function computes the expected number of edges in graphs sampled from
    the posterior approximation.
    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.
    Returns
    -------
    e_edges : float
        The expected number of edges.
    """
    num_edges = np.sum(posterior, axis=(1, 2))
    return np.mean(num_edges)


def threshold_metrics(posterior, ground_truth):
    """Compute threshold metrics (e.g. AUROC, Precision, Recall, etc...).
    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.
    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.
    Returns
    -------
    metrics : dict
        The threshold metrics.
    """
    # Expected marginal edge features
    p_edge = np.mean(posterior, axis=0)
    p_edge_flat = p_edge.reshape(-1)
    
    gt_flat = ground_truth.reshape(-1)

    # Threshold metrics 
    fpr, tpr, _ = metrics.roc_curve(gt_flat, p_edge_flat)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(gt_flat, p_edge_flat)
    prc_auc = metrics.auc(recall, precision)
    ave_prec = metrics.average_precision_score(gt_flat, p_edge_flat)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'prc_auc': prc_auc,
        'ave_prec': ave_prec,
    }

def log_likelihood_node_j(j, data, W, sigma=0.1):
    """
        Given a weighted adjacency matrix, and node index, calculate 
        the local log likelihood of data: P(X_j | G, theta)

        Parameters
        ----------
        j: int
            Node index for the local log likelihood calculation

        data: np.ndarray (n, d)
            Samples of causal variables.
        
        W: np.ndarray (d, d)
            Weighted adjacency matrix

        sigma: float
            Default standard deviation of the Normal distribution P(X_j | G, theta)

        Returns
        -------
        ll_j : float
            log P(X_j | G, theta)
    """
    N = data.shape[0]
    squared_error = -0.5 * np.sum((data[:, j] - (data @ W)[:, j]) ** 2) / (sigma ** 2)
    const_term = N * np.log(sigma) + (N / 2) * np.log(2 * np.pi)
    ll_j = squared_error - const_term
    return ll_j

def log_likelihood_per_g(data, W, sigma=0.1):
    """
        Given a weighted adjacency matrix, calculate log likelihood of data
        as sum of data log likelihood over each of the `d` nodes.

        Parameters
        ----------
        data: np.ndarray (n, d)
            Samples of causal variables.
        
        W: np.ndarray (d, d)
            Weighted adjacency matrix

        sigma: float
            Default standard deviation of the Normal distribution P(X | G, theta)

        Returns
        -------
        ll : float
            \sum_{j=0}^{d-1} log P(X_j | G, theta) 
    """
    d = W.shape[-1]
    ll = 0.
    for j in range(d):
        ll_j = log_likelihood_node_j(j, data, W, sigma)
        ll += ll_j
    return ll

def LL(gs, thetas, data, sigma):
    """
        Compute the observational log likelihood: P(X | G, theta)
        Parameters
        ----------
        gs : np.ndarray (num_graphs, d, d)
            Contains `num_graphs` adjacency matrices where `num_graphs`
            refers to number of posterior graph samples. `d` refers to 
            nodes in the graph.

        thetas : np.ndarray instance (num_graphs, d, d)
            Contains `num_graphs` weight matrices where `num_graphs`
            refers to number of posterior graph samples. `d` refers to 
            nodes in the graph.
        
        data: np.ndarray (n, d)
            Samples of causal variables. This is the X in P(X | G, theta)

        sigma: float or list
            Default standard deviation of the Normal distribution P(X | G, theta)

        Returns
        -------
        mean_log_likelihood : float
            log \mathbb{E}_{G, theta} P(X | G_i, theta_i)
    """
    log_likelihoods = []
    num_graphs, d, _ = gs.shape

    if isinstance(sigma, float):
        for g, theta in zip(gs, thetas):
            W = np.multiply(g, theta)
            log_likelihood = log_likelihood_per_g(data, W, sigma)
            log_likelihoods.append(log_likelihood) 
    else:
        for g, theta, sig in zip(gs, thetas, sigma):
            W = np.multiply(g, theta)
            log_likelihood = log_likelihood_per_g(data, W, sig)
            log_likelihoods.append(log_likelihood) 
    
    log_expected_likelihood = logsumexp(np.array(log_likelihoods)) - np.log(len(log_likelihoods))
    return log_expected_likelihood
