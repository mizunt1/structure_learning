import numpy as np
from numpy.random import default_rng
import networkx as nx

from dibs.target import make_linear_gaussian_model, make_nonlinear_gaussian_model
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood

from gflownet_sl.metrics.metrics import expected_shd, expected_edges, threshold_metrics, get_mean_and_ci
from gflownet_sl.utils.exhaustive import (get_full_posterior,
    get_edge_log_features, get_path_log_features, get_markov_blanket_log_features)
from gflownet_sl.utils.sampling import sample_from_linear_gaussian
from gflownet_sl.utils.graph import sample_erdos_renyi_linear_gaussian, sample_erdos_renyi_linear_gaussian_3_nodes
from gflownet_sl.utils.metrics import posterior_estimate, get_log_features

rng = default_rng(args.seed)
graph = sample_erdos_renyi_linear_gaussian(
    num_variables=args.num_variables,
    num_edges=args.num_edges,
    loc_edges=0.0,
    scale_edges=1.0,
    obs_noise=0.1,
    rng=rng,
    block_small_theta=args.block_small_theta
)
# specifies graph to sample from

data = sample_from_linear_gaussian(
    graph,
    num_samples=args.num_samples,
    rng=rng)
# samples from that graph

### results for results section 1 ###

_, model = make_linear_gaussian_model(key=rng, n_vars=20, graph_prior_str="sf")
dibs = JointDiBS(x=data.x, interv_mask=None, inference_model=model)
negll = neg_ave_log_likelihood(dist=dist, eltwise_log_likelihood=dibs.eltwise_log_likelihood_observ, x=data.x_ho)
# using some dibs code to calculate likelihood metrics

gt_adjacency = nx.to_numpy_array(graph, weight=None)
# adjacency of true graph
results = expected_shd(dist=est_posterior, g=gt_adjacency)
mean_shd, ci95_shd = get_mean_and_ci(*results, num_samples=est_posterior.shape[0], alpha=0.95)
# calc mean_shd , ci95 not important
thresholds = threshold_metrics(dist=est_posterior, g=gt_adjacency)
roc = thresholds.roc_auc


### results for results section 2 ###
full_posterior = get_full_posterior(
    data, score='lingauss', verbose=True, prior_mean=0., prior_scale=1., obs_scale=args.obs_noise)
# enumerates full posterior and all necessary features of full posterior


full_edge_log_features = list(get_edge_log_features(full_posterior).values())
full_path_log_features = list(get_path_log_features(full_posterior).values())
full_markov_log_features = list(get_markov_blanket_log_features(full_posterior).values())
# edge, path, markov features of full posterior as lists


est_log_features = get_log_features(est_posterior, data.columns)
# est_posterior is np.array of adjacency matrices of samples from the inferred (estimated) posterior.

est_edge = list(est_log_features.edge.values())
est_path = list(est_log_features.path.values())
est_markov = list(est_log_features.markov_blanket.values())
# get edge, path and markov features of estimated posteriors as lists

path_corr = np.corrcoef(np.exp(full_path_log_features), np.exp(est_path))[0][1]
# correlation coefficients, these should be close to one. Good to check.
markov_corr = np.corrcoef(np.exp(full_markov_log_features), np.exp(est_markov))[0][1]
edge_corr = np.corrcoef(np.exp(full_edge_log_features), np.exp(est_edge))[0][1]

# results for results section 3

