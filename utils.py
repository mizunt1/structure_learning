import numpy as np
def get_weighted_adjacency(graph):
    adjacency = np.zeros((len(graph), len(graph)), dtype=np.float_)
    index = dict((node, idx) for (idx, node) in enumerate(graph.nodes))
    for v in graph.nodes:
        cpd = graph.get_cpds(v)
        for u, weight in zip(cpd.evidence, cpd.mean[1:]):
            adjacency[index[u], index[v]] = weight
    return adjacency

def edge_marginal_means(means, adjacency_matrices):
    num_variables = adjacency_matrices.shape[1]
    num_matrices =  adjacency_matrices.shape[0]
    adjacency_expectation = np.sum(adjacency_matrices, axis=0)/num_matrices
    edge_marginal_means = np.zeros((num_variables, num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            edge_marginal_means[i][j] = adjacency_expectation[i][j] * means[i][j].item()
    return edge_marginal_means
