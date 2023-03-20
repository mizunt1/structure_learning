import numpy as np
import jax.numpy as jnp
from jax.ops import index, index_update
from jax import random
import networkx as nx

import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")


class InvalidCPDAGError(Exception):
    # raised when a "CPDAG" returned by a learning alg does not admit a random extension
    pass


class ContinualInvalidCPDAGError(Exception):
    # raised when a "CPDAG" returned by a learning alg does not admit a random extension
    # even after repeating several times with hard confidence thresholds
    pass


def nx_adjacency(g):
    return jnp.array(nx.adj_matrix(g).toarray())


def adjmat_to_str(mat, max_len=40):
    """
    Converts {0,1}-adjacency matrix to human-readable string
    """
    mat = np.asarray(mat)
    edges_mat = np.where(mat == 1)
    undir_ignore = set() # undirected edges, already printed

    def get_edges():
        for e in zip(*edges_mat):
            u, v = e
            # undirected?
            if mat[v, u] == 1:
                # check not printed yet
                if e not in undir_ignore:
                    undir_ignore.add((v, u))
                    yield (u, v, True)
            else:
                yield (u, v, False)

    strg = '  '.join([(f'{e[0]}--{e[1]}' if e[2] else
                       f'{e[0]}->{e[1]}') for e in get_edges()])
    if len(strg) > max_len:
        return strg[:max_len] + ' ... '
    elif strg == '':
        return '<empty graph>'
    else:
        return strg


def random_consistent_expansion(*, key, cpdag):
    '''
    Generates a "consistent extension" DAG of a CPDAG as defined by
    https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf
    i.e. a graph where DAG and CPDAG have the same skeleton and v-structures
    and every directed edge in the CPDAG has the same direction in the DAG

    This is achieved using the algorithm of
    http://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf

    Every DAG in the MEC is a consistent extension of the corresponding CPDAG.

    Arguments:
        key
        cpdag:  [n_vars, n_vars]
                adjacency matrix of a CPDAG;
                breaks if it is not a valid CPDAG (merely a PDAG)
                (i.e. if cannot be extended to a DAG, e.g. undirected ring graph)
    Returns:
        [n_vars, n_vars] : adjacency matrix of a DAG

    '''

    # check whether there are any undirected edges at all
    if jnp.sum(cpdag == cpdag.T) == 0:
        return cpdag

    G = cpdag.copy()
    A = cpdag.copy()

    N = A.shape[0]
    n_left = A.shape[0]
    node_exists = jnp.ones(A.shape[0])

    key, subk = random.split(key)
    ordering = random.permutation(subk, N)

    while n_left > 0:

        # find i satisfying:
        #   1) no directed edge leaving i (i.e. sink)
        #   2) undirected edge (i, j) must have j adjacent to all adjacent nodes of i
        #      (to avoid forming new v-structures when directing j->i)
        # If a valid CPDAG is input, then such an i must always exist, as every DAG in the MEC of a CPDAG is a consistent extension

        found_any_valid_candidate = False
        for i in ordering:

            if node_exists[i] == 0:
                continue

            # no outgoing _directed_ edges: (i,j) doesn't exist, or, (j,i) also does
            directed_i_out = A[i, :] == 1
            directed_i_in = A[:, i] == 1

            is_sink = jnp.all((1 - directed_i_out) | directed_i_in)
            if not is_sink:
                continue

                # for each undirected neighbor j of sink i
            i_valid_candidate = True
            undirected_neighbors_i = (directed_i_in == 1) & (directed_i_out == 1)
            for j in jnp.where(undirected_neighbors_i)[0]:

                # check that adjacents of i are a subset of adjacents j
                # i.e., check that there is no adjacent of i (ingoring j) that is not adjacent to j
                adjacents_j = (A[j, :] == 1) | (A[:, j] == 1)
                is_not_j = jnp.arange(N) != j
                if jnp.any(directed_i_in & (1 - adjacents_j) & is_not_j):
                    i_valid_candidate = False
                    break

                    # i is valid, orient all edges towards i in consistent extension
            # and delete i and all adjacent egdes
            if i_valid_candidate:
                found_any_valid_candidate = True

                # to orient G towards i, delete (oppositely directed) i,j edges from adjacency
                G = index_update(G, index[i, jnp.where(undirected_neighbors_i)], 0)

                # remove i in A
                A = index_update(A, index[i, :], 0)
                A = index_update(A, index[:, i], 0)

                node_exists = index_update(node_exists, index[i], 0)

                n_left -= 1

                break

        if not found_any_valid_candidate:
            err_msg = (
                    'found_any_valid_candidate = False; unable to create random consistent extension of CPDAG: ' + adjmat_to_str(
                cpdag) +
                    ' | G: ' + adjmat_to_str(G) +
                    ' | A: ' + adjmat_to_str(A) +
                    ' | ordering : ' + str(ordering.tolist())
            )
            raise InvalidCPDAGError(err_msg)

    return G
