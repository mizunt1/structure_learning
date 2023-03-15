import string 

import numpy as np
from numpy.random import default_rng

from gflownet_sl.utils.graph import sample_erdos_renyi_linear_gaussian

let_num = {letter: index for index, letter in enumerate(string.ascii_uppercase, start=0)} 

seed = 1
num_variables = 5
num_edges = 5
rng = default_rng(seed)
graph_ = sample_erdos_renyi_linear_gaussian(
    num_variables=num_variables,
    num_edges=num_edges,
    loc_edges=0.0,
    scale_edges=1.0,
    obs_noise=0.1,
    rng=rng
)
def graph_to_matrix(graph, num_nodes):
    matrix = np.zeros((num_nodes, num_nodes))
    for child in graph.adjacency():
        # adj shows child of node
        for n, parent in enumerate(graph.get_parents(child[0])):        
            theta = graph.get_cpds(child[0]).mean[n+1]
            matrix[let_num[parent]][let_num[child[0]]] = theta
    return matrix
def get_weighted_adjacency(graph):
    adjacency = np.zeros((len(graph), len(graph)), dtype=np.float_)
    index = dict((node, idx) for (idx, node) in enumerate(graph.nodes))
    for v in graph.nodes:
        cpd = graph.get_cpds(v)
        for u, weight in zip(cpd.evidence, cpd.mean[1:]):
            adjacency[index[u], index[v]] = weight
    return adjacency

