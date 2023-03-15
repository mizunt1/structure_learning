import numpy as np
import networkx as nx

from scipy.special import logsumexp
from dataclasses import dataclass
from collections import namedtuple, defaultdict
from pgmpy.estimators import ExhaustiveSearch, BDeuScore
from tqdm import tqdm

from gflownet_sl.utils.graph import get_markov_blanket_graph
from gflownet_sl.scores.pgmpy_bge_score import BGeScore
from gflownet_sl.scores.lingauss import LinearGaussianScore

# https://oeis.org/A003024
NUM_DAGS = [1, 1, 3, 25, 543, 29281, 3781503]

class GraphCollection:
    def __init__(self):
        self.edges, self.lengths = [], []
        self.mapping = defaultdict(int)
        self.mapping.default_factory = lambda: len(self.mapping)

    def append(self, graph):
        self.edges.extend([self.mapping[edge] for edge in graph.edges()])
        self.lengths.append(graph.number_of_edges())

    def freeze(self):
        self.edges = np.asarray(self.edges, dtype=np.int_)
        self.lengths = np.asarray(self.lengths, dtype=np.int_)
        self.mapping = [edge for (edge, _)
            in sorted(self.mapping.items(), key=lambda x: x[1])]
        return self

    def is_frozen(self):
        return isinstance(self.mapping, list)
    
    def to_dict(self, prefix=None):
        prefix = f'{prefix}_' if (prefix is not None) else ''
        return ({
            f'{prefix}edges': self.edges,
            f'{prefix}lengths': self.lengths,
            f'{prefix}mapping': self.mapping
        })

    def load(self, edges, lengths, mapping):
        self.edges = edges
        self.lengths = lengths
        self.mapping = dict((tuple(edge), idx) for (idx, edge) in enumerate(mapping))
        return self.freeze()


@dataclass
class FullPosterior:
    log_probas: np.ndarray
    graphs: GraphCollection
    closures: GraphCollection
    markov: GraphCollection

    def to_dict(self):
        # Ensure that "graphs" has been frozen
        if not self.graphs.is_frozen():
            raise ValueError('Graphs must be frozen. Call "graphs.freeze()".')

        offset, output = 0, dict()
        for length, log_prob in zip(self.graphs.lengths, self.log_probas):
            edges_indices = self.graphs.edges[offset:offset + length]
            edges = [self.graphs.mapping[idx] for idx in edges_indices]
            output[frozenset(edges)] = log_prob
            offset += length

        return output

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f, log_probas=self.log_probas,
                **self.graphs.to_dict(prefix='graphs'),
                **self.closures.to_dict(prefix='closures'),
                **self.markov.to_dict(prefix='markov')
            )

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            log_probas = data['log_probas']
            graphs = GraphCollection().load(
                data['graphs_edges'],
                data['graphs_lengths'],
                data['graphs_mapping']
            )
            closures = GraphCollection().load(
                data['closures_edges'],
                data['closures_lengths'],
                data['closures_mapping']
            )
            markov = GraphCollection().load(
                data['markov_edges'],
                data['markov_lengths'],
                data['markov_mapping']
            )
        return cls(
            log_probas=log_probas,
            graphs=graphs,
            closures=closures,
            markov=markov
        )

def get_full_posterior(data, score='bdeu', verbose=True, **kwargs):
    if score == 'bdeu':
        scoring_method = BDeuScore(data, **kwargs)
    elif score == 'bge':
        scoring_method = BGeScore(data, **kwargs)
    elif score == 'lingauss':
        scoring_method = LinearGaussianScore(data, **kwargs)
    else:
        raise NotImplementedError('Only "bdeu" and "bge" implemented.')
    estimator = ExhaustiveSearch(data, scoring_method=scoring_method)

    log_probas = []
    graphs = GraphCollection()
    closures = GraphCollection()
    markov = GraphCollection()
    with tqdm(estimator.all_dags(), 
            total=NUM_DAGS[data.shape[1]], disable=(not verbose)) as pbar:
        for graph in pbar:  # Enumerate all possible DAGs
            score = estimator.scoring_method.score(graph)
            log_probas.append(score)

            graphs.append(graph)
            closures.append(nx.transitive_closure_dag(graph))
            markov.append(get_markov_blanket_graph(graph))

    # Normalize the log-joint distribution to get the posterior
    log_probas = np.asarray(log_probas, dtype=np.float_)
    log_probas -= logsumexp(log_probas)

    return FullPosterior(
        log_probas=log_probas,
        graphs=graphs.freeze(),
        closures=closures.freeze(),
        markov=markov.freeze()
    )


def _get_log_features(graphs, log_probas):
    indices = np.zeros_like(graphs.lengths)
    indices[1:] = np.cumsum(graphs.lengths[:-1])

    features = dict()
    for index, edge in enumerate(graphs.mapping):
        if not np.any(graphs.edges == index):
            continue
        has_feat = np.add.reduceat(graphs.edges == index, indices)

        # Edge case: the first graph is the empty graph, it has no edge
        if graphs.lengths[0] == 0:
            has_feat[0] = 0
        assert np.sum(graphs.edges == index) == np.sum(has_feat)
        
        has_feat = has_feat.astype(np.bool_)
        features[edge] = logsumexp(log_probas[has_feat])

    return features

def get_edge_log_features(posterior):
    return _get_log_features(posterior.graphs, posterior.log_probas)


def get_path_log_features(posterior):
    return _get_log_features(posterior.closures, posterior.log_probas)


def get_markov_blanket_log_features(posterior):
    return _get_log_features(posterior.markov, posterior.log_probas)


if __name__ == '__main__':
    from pgmpy.utils import get_example_model
    from gflownet_sl.utils.sampling import sample_from_discrete

    model = get_example_model('cancer')
    samples = sample_from_discrete(model, num_samples=100)

    posterior = get_full_posterior(samples)
    features = get_markov_blanket_log_features(posterior)
    print(features)