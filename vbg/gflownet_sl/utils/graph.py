import numpy as np
import networkx as nx
import string
import pandas as pd

from itertools import chain, product, islice, count

from numpy.random import default_rng, binomial, uniform
from pgmpy import models
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.estimators import ExhaustiveSearch

def _s(node1, node2):
    return (node2, node1) if (node1 > node2) else (node1, node2)

def get_markov_blanket(graph, node):
    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))

    mb_nodes = parents | children
    for child in children:
        mb_nodes |= set(graph.predecessors(child))
    mb_nodes.discard(node)

    return mb_nodes


def get_markov_blanket_graph(graph):
    """Build an undirected graph where two nodes are connected if
    one node is in the Markov blanket of another.
    """
    # Make it a directed graph to control the order of nodes in each
    # edges, to avoid mapping the same edge to 2 entries in mapping.
    mb_graph = nx.DiGraph()
    mb_graph.add_nodes_from(graph.nodes)

    edges = set()
    for node in graph.nodes:
        edges |= set(_s(node, mb_node)
            for mb_node in get_markov_blanket(graph, node))
    mb_graph.add_edges_from(edges)

    return mb_graph


def adjacencies_to_networkx(adjacencies, nodes):
    mapping = dict(enumerate(nodes))
    for adjacency in adjacencies:
        graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
        yield nx.relabel_nodes(graph, mapping, copy=False)


def sample_erdos_renyi_graph(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        create_using=models.BayesianNetwork,
        rng=default_rng()
    ):
    if p is None:
        if num_edges is None:
            raise ValueError('One of p or num_edges must be specified.')
        p = num_edges / ((num_variables * (num_variables - 1)) / 2.)
    
    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1))
        nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]

    graph = nx.from_numpy_array(adjacency, create_using=create_using)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph


def sample_erdos_renyi_linear_gaussian(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        loc_edges=0.0,
        scale_edges=1.0,
        low_edges = 0.5,
        obs_noise=0.1,
        rng=default_rng(),
        block_small_theta=False
    ):
    # Create graph structure
    graph = sample_erdos_renyi_graph(
        num_variables,
        p=p,
        num_edges=num_edges,
        nodes=nodes,
        create_using=models.LinearGaussianBayesianNetwork,
        rng=rng
    )
    # Create the model parameters
    factors = []
    for node in graph.nodes:
        parents = list(graph.predecessors(node))

        # Sample random parameters (from Normal distribution)
        if block_small_theta:
            pos_or_neg = (-1)**(rng.binomial(1, 0.5, size=(len(parents)+1,)))
            value = rng.uniform(low=low_edges, high=scale_edges, size=(len(parents)+1,))
            theta = pos_or_neg*value
                                
        else:
            theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
        theta[0] = 0.  # There is no bias term

        # Create factor
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    graph.add_cpds(*factors)
    return graph

def sample_erdos_renyi_linear_gaussian_3_nodes(
        graph_index,
        p=None,
        num_edges=None,
        nodes=['A', 'B', 'C'],
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=0.1,
        rng=default_rng(),
        block_small_theta=False
    ):
    # Create graph structure
    graphs = ExhaustiveSearch(pd.DataFrame(np.random.randint(2,3, size=(10,3)), columns=list('ABC')))
    dags = list(graphs.all_dags())
    graph = dags[graph_index]
    adjacency = nx.adjacency_matrix(graph, nodelist=['A', 'B', 'C']).toarray()

    graph = nx.from_numpy_array(adjacency, create_using=models.LinearGaussianBayesianNetwork)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)
    # Create the model parameters
    factors = []
    for node in graph.nodes:
        parents = list(graph.predecessors(node))

        # Sample random parameters (from Normal distribution)
        if block_small_theta:
            pos_or_neg = (-1)**(rng.binomial(1, 0.5, size=(len(parents)+1,)))
            value = rng.uniform(low=0.7, high=2, size=(len(parents)+1,))
            theta = pos_or_neg*value
                                
        else:
            theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
        theta[0] = 0.  # There is no bias term

        # Create factor
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    graph.add_cpds(*factors)


    return graph


def get_weighted_adjacency(graph):
    adjacency = np.zeros((len(graph), len(graph)), dtype=np.float_)
    index = dict((node, idx) for (idx, node) in enumerate(graph.nodes))
    for v in graph.nodes:
        cpd = graph.get_cpds(v)
        for u, weight in zip(cpd.evidence, cpd.mean[1:]):
            adjacency[index[u], index[v]] = weight
    return adjacency


if __name__ == '__main__':
    from pgmpy.utils import get_example_model

    graph = get_example_model('cancer')
    mb_graph = get_markov_blanket_graph(graph)
    print(mb_graph.edges())

    graph = sample_erdos_renyi_linear_gaussian(5, num_edges=2, nodes='ABCDE')
    print(graph.edges)
