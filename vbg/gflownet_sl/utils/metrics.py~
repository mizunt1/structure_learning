import jax.numpy as jnp
import haiku as hk
import numpy as np
import math
import networkx as nx
import os 

from jax import random, vmap, lax, jit, nn
from functools import partial
from tqdm import tqdm, trange
from collections import namedtuple, defaultdict

from gflownet_sl.nets.gflownet import gflownet
from gflownet_sl.utils.gflownet import log_policy
from gflownet_sl.utils.jnp_utils import batch_random_choice
from gflownet_sl.utils.graph import adjacencies_to_networkx, get_markov_blanket_graph


def update_bwd(adjacency, closure, action):
    num_variables = adjacency.shape[0]
    source, target = jnp.divmod(action, num_variables)
    
    adjacency = adjacency.at[source, target].set(0.)
    closure = closure - jnp.outer(closure[:, target], closure[source])
    return (adjacency, closure)


def get_transitive_closure(adjacency):
    # Warshall's algorithm
    def body_fun(i, closure):
        outer_product = jnp.outer(closure[:, i], closure[i])
        return jnp.logical_or(closure, outer_product)

    adjacency = adjacency.astype(jnp.bool_)
    closure = lax.fori_loop(0, adjacency.shape[0], body_fun, adjacency)
    closure = closure.at[jnp.diag_indices_from(closure)].set(True)
    return closure.astype(jnp.float32)


@partial(jit, static_argnums=(3,))
def log_likelihood_estimate(params, key, adjacency, num_samples=100):
    """
    We make the assumption that the number of parents in the graph
    corresponding to "adjacency" is, for all the nodes, smaller than the
    maximum number of parents allowed in the environment. Otherwise, the graph
    would not be reachable in the environment, and we wouldn't be able to
    compute its log-likelihood.
    """
    model = hk.transform(gflownet)
    vmodel = vmap(model.apply, in_axes=(None, None, 0, None))

    def cond_fun(state):
        adjacency, *_ = state
        return jnp.any(adjacency)

    def body_fun(state):
        adjacency, closure, key, log_likelihoods = state

        # Create the uniform distribution over previous states
        flat_adjacency = adjacency.reshape(num_samples, -1)
        num_edges = jnp.sum(flat_adjacency, axis=1, keepdims=True)

        # Sample actions
        key, subkey = random.split(key)
        masks = 1. - (adjacency + closure)
        actions = batch_random_choice(subkey, flat_adjacency / num_edges, masks)

        # Update the adjacency matrix & transitive closure
        adjacency, closure = vmap(update_bwd)(adjacency, closure, actions)

        # Compute the log probabilities
        outputs = vmodel(params, subkey, adjacency, False)
        log_probas = log_policy(outputs, 1. - (adjacency + closure))
        log_probas = jnp.take_along_axis(log_probas, actions[:, None], axis=1)

        # Update the log-likelihood
        log_likelihoods = log_likelihoods + log_probas

        return (adjacency, closure, key, log_likelihoods)

    closure = get_transitive_closure(adjacency.T)
    log_norm = -lax.lgamma(1. + jnp.sum(adjacency))
    adjacency = jnp.repeat(adjacency[None], num_samples, axis=0)
    closure = jnp.repeat(closure[None], num_samples, axis=0)

    outputs = vmodel(params, key, adjacency, False)
    init_state = (adjacency, closure, key, nn.log_sigmoid(outputs.stop))
    *_, log_likelihoods = lax.while_loop(cond_fun, body_fun, init_state)

    return nn.logsumexp(log_likelihoods) - math.log(num_samples) - log_norm


@jit
def sample_action_from_gflownet(params, key, observations):
    model = hk.transform(gflownet)
    vmodel = vmap(model.apply, in_axes=(None, None, 0, None))
    masks = observations['mask'].astype(jnp.float32)

    # Compute the log probabilities
    outputs = vmodel(params, key,
        observations['adjacency'].astype(jnp.float32), False)
    log_probas = log_policy(outputs, masks)

    # Sample actions
    return batch_random_choice(key, jnp.exp(log_probas), masks)


def posterior_estimate(params, env, key, num_samples=100, verbose=True):
    samples = []
    observations = env.reset()
    with trange(num_samples, disable=(not verbose)) as pbar:
        while len(samples) < num_samples:
            order = observations['order']
            key, subkey = random.split(key)
            actions = sample_action_from_gflownet(params, subkey, observations)
            observations, _, dones, _ = env.step(np.asarray(actions))

            samples.extend([order[i] for i, done in enumerate(dones) if done])
            pbar.update(min(num_samples - pbar.n, np.sum(dones).item()))
    return np.stack(samples[:num_samples], axis=0)


Features = namedtuple('Features', ['edge', 'path', 'markov_blanket'])

def get_log_features(posterior, nodes, verbose=True):
    """Compute the log-features for edges, paths & Markov blankets."""
    features = Features(
        edge=defaultdict(float),
        path=defaultdict(float),
        markov_blanket=defaultdict(float)
    )
    num_samples = posterior.shape[0]
    for graph in tqdm(adjacencies_to_networkx(posterior, nodes),
            total=num_samples, disable=(not verbose)):
        # Get edge features
        for edge in graph.edges:
            features.edge[edge] += 1.

        # Get path features
        closure = nx.transitive_closure_dag(graph)
        for edge in closure.edges:
            features.path[edge] += 1.

        # Get Markov blanket features
        mb = get_markov_blanket_graph(graph)
        for edge in mb.edges:
            features.markov_blanket[edge] += 1.

    return Features(
        edge=dict((key, math.log(value) - math.log(num_samples))
            for (key, value) in features.edge.items()),
        path=dict((key, math.log(value) - math.log(num_samples))
            for (key, value) in features.path.items()),
        markov_blanket=dict((key, math.log(value) - math.log(num_samples))
            for (key, value) in features.markov_blanket.items())
    )

def return_file_paths(seed, result, method, base_dir=os.path.join(
        "/network","scratch",
        "m","mizu.nishikawa-toomey",
        "gflowdag")):
    """
    seed (int) : 
    result (str): either "results1" or "results2"
    method (str): specify the method type
    """
    summary_method = os.path.join(base_dir, result, "summary" + "_" + method + ".csv")
    summary_all_data = os.path.join(base_dir, result, "summary_all_data" + "_" + method + ".csv")
    # summary of results for whole method, takes mean and c95 over all seeds.
    base_dir = os.path.join(base_dir,
                            result,
                            "seed"+str(seed))
    path_name_data = os.path.join(base_dir, "data.csv")
    # path to training data
    path_name_data_test = os.path.join(base_dir, "data_test.csv")
    # path to test data
    path_name_graph = os.path.join(base_dir, "graph.pkl")
    # path to saved ground truth graph
    if method == "data_res1":
        return {"data": path_name_data, "data_test": path_name_data_test,
                "graph": path_name_graph}
    elif method == "data_res2":
        path_name_true_post = os.path.join(base_dir, "true_posterior.pkl")
        return {"data": path_name_data, "data_test": path_name_data_test,
                "graph": path_name_graph, "true_post": path_name_true_post}
            
    else:
        path_name_true_post = os.path.join(base_dir, "true_posterior.pkl")
        # path to the pkl of the full posterior. For graphs smaller than 5 only
        
        base_dir = os.path.join(base_dir, method)
        path_name_results = os.path.join(base_dir, "results.csv")
        # result of csv contains SHD, AUROC, correlation coeffs etc for one method, and one seed.
        path_name_true_post_bcd = os.path.join(base_dir, "true_posterior_bcd.pkl")
        # path to the pkl of the full posterior. For graphs smaller than 5 only
        path_name_sigma_bcd = os.path.join(base_dir, "posterior_Sigma.npy")
        # path to the pkl of the full posterior. For graphs smaller than 5 only
        
        path_name_model = os.path.join(base_dir, "model.npz")
        bcd_posterior = os.path.join(base_dir, "bcd_post.pkl")
        # path to trained weights of model if they exist
        path_name_edge =  os.path.join(base_dir,  "edge.csv")
        # full and estimated edge features for each pair of nodes
        path_name_markov =  os.path.join(base_dir, "markov.csv")
        # full and estimated markov features for each pair of nodes
        path_name_path =  os.path.join(base_dir, "path.csv")
        # full and estimated path features for each pair of nodes
        path_name_theta_params = os.path.join(base_dir, "theta_params.pkl")
        # parameters that parameterise the approximate posterior. Means, covariances etc
        # can be any python object
        path_name_est_post_g = os.path.join(base_dir, "posterior_estimate.npy")
        # samples of the posterior over gs
        path_name_est_post_theta = os.path.join(base_dir, "posterior_estimate_thetas.npy")
        # samples of the posterior over thetas
        return {"data": path_name_data, "data_test": path_name_data_test,
                "graph": path_name_graph, "results": path_name_results,
                "model": path_name_model, "edge": path_name_edge,
                "markov": path_name_markov, "path": path_name_path,
                "true_post": path_name_true_post, "est_post_g": path_name_est_post_g,
                "theta_params": path_name_theta_params,
                "est_post_theta": path_name_est_post_theta, "summary": summary_method,
                "true_post_bcd": bcd_posterior,
                "sigma_bcd": path_name_sigma_bcd,
                "summary_all_data": summary_all_data}

if __name__ == '__main__':
    model = hk.transform(gflownet)
    rng = hk.PRNGSequence(0)

    adjacency = jnp.zeros((3, 3))
    params = model.init(next(rng), adjacency, False)

    adjacency = jnp.array([
        [0., 1., 0.],
        [0., 0., 0.],
        [1., 0., 0.]
    ])

    log_likelihood = log_likelihood_estimate(params, next(rng), adjacency, num_samples=2)
    print(log_likelihood)

    # from pgmpy.utils import get_example_model
    # from gflownet_sl.env import GFlowNetDAGEnv
    # from gflownet_sl.utils.sampling import sample_from_discrete

    # graph = get_example_model('sachs')
    # samples = sample_from_discrete(graph, num_samples=1000)

    # env = GFlowNetDAGEnv(num_envs=4, data=samples, max_parents=2, num_workers=4, context='spawn')
    # adjacency = jnp.zeros((env.num_variables, env.num_variables))
    # params = model.init(next(rng), adjacency, False)

    # probas = posterior_edge_features_estimate(params, env, next(rng), num_samples=10)
    # print(probas)

        
     
                                            
    
