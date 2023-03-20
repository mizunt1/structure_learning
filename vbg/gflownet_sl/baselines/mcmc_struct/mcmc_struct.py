import time
import numpy as np
import igraph as ig
from tqdm.auto import trange
from datetime import datetime

import jax.numpy as jnp
from jax import random

# for BaselineStructureMCMC
from gflownet_sl.baselines.mcmc_struct.linearGaussianGaussianEquivalent import BGe
from gflownet_sl.baselines.mcmc_struct.graphdistributions import UniformDAGDistributionRejection, ErdosReniDAGDistribution
from gflownet_sl.scores import BDeuScore


# NO NOT USE THIS CLASS! USE INSTEAD THE SUBCLASS WrapperStructureMCMC
class StructureMCMC:
    """
    Structure MCMC as by Giudici and Castelo (2003)

        only_non_covered: boolean indicating whether only non-covered edge reversals should be considered

    """

    def __init__(self, *, n_vars, only_non_covered=False, verbose=True):
        super(StructureMCMC, self).__init__()

        self.n_vars = n_vars
        self.only_non_covered = only_non_covered
        self.verbose = verbose

    def topsort_descendants(self, *, inc, source):
        """
        Returns topological sorting of all descendants of `source` in
        graph with incidency matrix `inc`.
        Thus, returns list of the form:
            [source, ...]

        """
        visited = set()
        rev_toporder = []

        def dfs(u):
            if u in visited:
                return
            visited.add(u)
            for v in np.where(inc[:, u] == 1)[0]:
                dfs(v)
            rev_toporder.append(u)

        dfs(source)

        return rev_toporder[::-1]

    def is_valid_add(self, *, i, j, g, inc, anc):
        """
        Returns `True` iff adding edge i -> j does not create a cycle.
        """

        # if j ~> ... ~> i exists, then edge i -> j forms cycle
        valid = (anc[i, j] == 0)

        # if desired, can check for fan-in constraint here (i.e. max no. of parents)

        return valid

    def is_valid_reverse(self, *, i, j, g, inc, anc):
        """
        Returns `True` iff reversing edge i -> j to j -> i does not create a cycle.
        """
        # if any parent k of j other than i has i as ancestor, j -> i forms a cycle
        self.n_vars = inc.shape[0]
        for k in range(self.n_vars):
            if (inc[j, k] == 1) and (not i == k):
                if anc[k, i] == 1:
                    return False

        # check that the edge-reversal is `non-covered`
        # i.e. the reversal must move to a new equivalence class
        # (an edge is covered if on its removal the parent sets of the two connected nodes are identical)
        if self.only_non_covered:
            par_i, par_j = inc[i], inc[j]
            mask = (np.arange(self.n_vars) != i)
            covered = np.all(par_i[mask] == par_j[mask])
            if not covered:
                return False

        return True

    def get_valid_moves(self, *, g, inc, anc):
        """
        Returns list of igraph.Graph objects forming the `neighborhood` of g,
        meaning all graphs that are within single edge (i) additions, (ii) removals, or (iii) reversals
        from g and remain acyclic

            g: igraph.Graph
            inc: [n_vars, n_vars] incidence matrix
            anc: [n_vars, n_vars] ancestor matrix

        """

        moves = []

        # iterate over all possible edges i -> j edges in graph
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if i == j:
                    continue

                # check whether i -> j exists
                # if yes: remove and try reverse; if no, try add
                if inc[j, i] == 1:
                    # remove always valid
                    moves.append(('rem', i, j))

                    # try reverse
                    if self.is_valid_reverse(i=i, j=j, g=g, inc=inc, anc=anc):
                        moves.append(('rev', i, j))

                else:
                    # try add
                    if self.is_valid_add(i=i, j=j, g=g, inc=inc, anc=anc):
                        moves.append(('add', i, j))

        return moves

    def apply_move(self, *, g, m):
        """
        Applies move `m` to graph `g`
            g: `igraph.Graph`
            m: 3-tuple ('add/rem/rev', source, target)
        Returns:
            g_prime
        """

        mode, source, target = m

        if mode == 'add':
            g.add_edges([(source, target)])

        elif mode == 'rem':
            g.delete_edges([(source, target)])

        elif mode == 'rev':
            g.delete_edges([(source, target)])
            g.add_edges([(target, source)])
        else:
            raise ValueError('Invalid move.')

        return g

    def apply_add(self, *, i, j, inc, anc):
        """
        Changes incidence and ancestor matrix when adding
        edge i -> j
            i: edge source
            j: edge target
            inc: [n, n] incidence matrix
            anc: [n, n] ancestor matrix
        """

        assert (inc[j, i] == 0)
        inc[j, i] = 1

        # set i as ancestor of j
        anc[j, i] = 1

        # set all ancestors of i as ancestors of j
        anc[j] = np.logical_or(anc[j], anc[i])

        # set ancestors of j as ancestors of all descendants of j
        # descendants of j are all k s.t. j is ancestor (A[k, j] == 1)
        for k in range(self.n_vars):
            if anc[k, j] == 1:
                anc[k] = np.logical_or(anc[k], anc[j])

        return inc, anc

    def apply_remove(self, *, i, j, inc, anc):
        """
        Changes incidence and ancestor matrix when remvoing
        edge i -> j
            i: edge source
            j: edge target
            inc: [n, n] incidence matrix
            anc: [n, n] ancestor matrix
        """

        assert (inc[j, i] == 1)
        inc[j, i] = 0

        # need to rebuild ancestor matrix for j and descendants
        # in topological (DFS) order
        toporder = self.topsort_descendants(inc=inc, source=j)
        for u in toporder:
            # ancestors of u are incident vertices in updated I
            anc[u] = inc[u]

            # any ancestor of a parent of u is also an ancestor of u
            anc_parents = anc[np.where(inc[u] == 1)]  # ancestor arrays of each parent of u
            anc_parents_any = np.any(anc_parents, axis=0)  # whether of not any parent has j-th node as an ancestor
            anc[u] = np.logical_or(anc[u], anc_parents_any)  # make each ancestor of parent an ancestor of u

        return inc, anc

    def apply_mat_move(self, *, inc, anc, m):
        """
        Applies move `m` to graph incidence and ancestor matrices
            inc: [n, n] incidence matrix
            anc: [n, n] ancestor matrix
            m: 3-tuple ('add/rem/rev', source, target)
        Returns:
            g_prime
        """

        n = inc.shape[0]
        mode, source, target = m

        # apply the change
        if mode == 'add':
            inc, anc = self.apply_add(i=source, j=target, inc=inc, anc=anc)

        elif mode == 'rem':
            inc, anc = self.apply_remove(i=source, j=target, inc=inc, anc=anc)

        elif mode == 'rev':
            inc, anc = self.apply_remove(i=source, j=target, inc=inc, anc=anc)
            inc, anc = self.apply_add(i=target, j=source, inc=inc, anc=anc)

        else:
            raise ValueError('Invalid move.')

        return inc, anc

    def sample(self, *, key, n_samples, unnormalized_log_prob=None, unnormalized_log_prob_single=None, burnin=1e4,
               thinning=10, g_init=None,
               verbose_indication=0, return_matrices=True):
        """
        Generates `n_samples` graph samples  after `burning` steps, sampling every `thinning` steps.
        If `g_init` is not provided, starts with empty graph.

        unnormalized_log_prob: takes igraph.Graph as input, returns unnormalized log p(G) of the
            target distribution we would like to sample from

        """
        last_verbose_indication = 1
        t_start = time.time()

        modular_score = unnormalized_log_prob_single is not None

        # I[i, j] = 1 iff directed edge j -> i exists
        # A[i, j] = 1 iff there exists a directed path j ~> ... ~> i
        inc = np.zeros((self.n_vars, self.n_vars))
        anc = np.zeros((self.n_vars, self.n_vars))
        g = ig.Graph.Weighted_Adjacency(inc.T.tolist())

        if g_init is not None:
            # build matrices for provided graph
            g = g_init
            for e in g.es:
                i, j = e.tuple
                inc, anc = self.apply_mat_move(inc=inc, anc=anc, m=('add', i, j))

        if not modular_score:
            log_prob = unnormalized_log_prob(g)

        # assert(g.is_dag())

        moves = None
        graphs = []
        n_steps = int(burnin + thinning * n_samples)
        titer = trange(n_steps, desc='StructureMCMC', disable=not self.verbose)

        for t in titer:

            # compute valid moves (neighborhood)
            if moves is None:
                moves = self.get_valid_moves(g=g, inc=inc, anc=anc)

            # propose move
            key, subk = random.split(key)
            u = random.uniform(subk)

            key, subk = random.split(key)
            i = random.choice(subk, len(moves))

            move = moves[i]

            # bayes factor
            g_new = self.apply_move(g=g.copy(), m=move)

            if modular_score:
                # evaluate score only when parent set changed
                if move[0] == 'add':
                    bayes_factor = (
                            unnormalized_log_prob_single(g_new, move[2])
                            - unnormalized_log_prob_single(g, move[2])
                    )
                elif move[0] == 'rem':
                    bayes_factor = (
                            unnormalized_log_prob_single(g_new, move[2])
                            - unnormalized_log_prob_single(g, move[2])
                    )
                elif move[0] == 'rev':
                    bayes_factor = (
                            unnormalized_log_prob_single(g_new, move[1])
                            + unnormalized_log_prob_single(g_new, move[2])
                            - unnormalized_log_prob_single(g, move[1])
                            - unnormalized_log_prob_single(g, move[2])
                    )
                else:
                    raise ValueError('Invalid move')

            else:
                # evaluate score for full parent set
                log_prob_new = unnormalized_log_prob(g_new)
                bayes_factor = log_prob_new - log_prob

            # assert(g_new.is_dag())

            # metropolis hastings
            alpha = min(0, bayes_factor)
            if np.log(u) < alpha:

                # accept move
                g = g_new
                inc, anc = self.apply_mat_move(inc=inc, anc=anc, m=move)
                moves = None
                if not modular_score:
                    log_prob = log_prob_new

            # collect samples
            if not t % thinning and t >= burnin:
                graphs.append(g.copy())

            # verbose progress
            if verbose_indication > 0:
                if t >= (last_verbose_indication * n_steps // verbose_indication):
                    print(
                        f'StructureMCMC   {t} / {n_steps} [{(100 * t / n_steps):3.1f} %' +
                        f'| {((time.time() - t_start) / 60):.0f} min | {datetime.now().strftime("%d/%m %H:%M")}]',
                        flush=True
                    )
                    last_verbose_indication += 1

        # randomly permute
        assert (len(graphs) == n_samples)
        key, subk = random.split(key)
        perm = random.permutation(subk, n_samples)
        graphs = [graphs[i] for i in perm]

        # convert to matrices
        if return_matrices:
            mats = []
            for g_ in graphs:
                mats.append(np.array(g_.get_adjacency().data))
            return jnp.array(mats)

        else:
            return graphs


class WrapperStructureMCMC(StructureMCMC):
    def __init__(self, *,
            data,
            only_non_covered=False,
            verbose=True,
            mean_obs=None,
            alpha_mu=None,
            alpha_lambd=None,
            discrete_interv=False,
            equivalent_sample_size=10,
            prior='uniform',
            beta=0.1,
            n_edges_per_node=1
        ):
        self.data = np.asarray(data)
        self.n_vars = self.data.shape[1]
        self.mean_obs = mean_obs or np.zeros(self.n_vars)
        self.alpha_mu = alpha_mu or 1.0
        self.alpha_lambd = alpha_lambd or (self.n_vars + 2)
        self.prior = prior
        self.discrete_interv = discrete_interv
        if self.discrete_interv:
            # discard info about interventions
            self.data = self.data[:, :-1]
            self.n_vars -= 1

        super().__init__(
            n_vars=self.n_vars,
            only_non_covered=only_non_covered,
            verbose=verbose
        )

        if not self.discrete_interv:
            if self.prior == 'uniform':
                g_dist = UniformDAGDistributionRejection(n_vars=self.n_vars)
            elif self.prior == 'erdos_renyi':
                g_dist = ErdosReniDAGDistribution(n_vars=self.n_vars)
            self._inference_model = BGe(
                g_dist=g_dist,
                mean_obs=self.mean_obs,
                alpha_mu=self.alpha_mu,
                alpha_lambd=self.alpha_lambd
            )
        else:
            self._inference_model = BDeuScore(data,
                                              equivalent_sample_size=equivalent_sample_size,
                                              has_interventional=True,
                                              prior=prior,
                                              beta=beta,
                                              n_edges_per_node=n_edges_per_node)

    def unnormalized_log_prob_single(self, g, j):
        if not self.discrete_interv:
            return self._inference_model.g_dist.unnormalized_log_prob_single(g=g, j=j) + \
                self._inference_model.log_marginal_likelihood_given_g_single(g=g, x=self.data, j=j)
        else:
            parent_edges = g.incident(j, mode='in')
            parents = tuple(g.es[e].source for e in parent_edges)
            _, score_ = self._inference_model.get_local_scores(j, parents)
            return score_.score
    
    def sample(self, *, key, n_samples, burnin=1e4, thinning=10):
        return super().sample(
            key=key,
            n_samples=n_samples,
            unnormalized_log_prob_single=self.unnormalized_log_prob_single,
            burnin=burnin,
            thinning=thinning,
            return_matrices=True
        )


if __name__ == '__main__':
    from gflownet_sl.metrics.metrics import expected_shd

    import jax
    key = jax.random.PRNGKey(0)

    # n_vars = 3
    # # data
    # data = np.array([[3.14, -2.041, -0.13],
    #                  [6.12, -4.083, -0.15],
    #                  [3.13, -2.039, 0.21],
    #                  [6.12, -4.082, 0.09],
    #                  [3.11, -2.040, 0.11]])

    # from eval_baselines import generate_data
    # n_vars = 20
    # degree = 4
    # path = './dags_n{}d{}'.format(n_vars, degree)
    # data, w_adjacency = generate_data(degree, n_vars, path=path)
    # binary_gtruth = (w_adjacency != 0).astype(float)

    import pandas as pd
    import networkx as nx
    from gflownet_sl.utils.interventional_sachs import download_sachs_intervention
    from pgmpy.utils import get_example_model
    gt_graph = get_example_model('sachs')
    gt_adjacency = nx.to_numpy_array(gt_graph, weight=None)
    download_sachs_intervention()
    data = pd.read_csv(
        'data/sachs.interventional.txt',
        delimiter=' ',
        dtype='category'
    )

    structMCMC = WrapperStructureMCMC(data=data,
                                      discrete_interv=True,
                                      prior='edge',
                                      beta=0.1)

    n_samples = 1000  # 30  # number of posterior samples to return
    # burnin = 10  # 100_000
    # thinning = 2  # 10_000

    structMCMC_samples = structMCMC.sample(key=key,
                                           n_samples=n_samples)  # ,
                                           # burnin=burnin,
                                           # thinning=thinning)

    print("Computing SHD for MCMC Structure:")
    print(expected_shd(dist=structMCMC_samples, g=gt_adjacency), flush=True)



    from gflownet_sl.metrics.metrics import expected_shd, expected_edges, threshold_metrics, get_mean_and_ci

    orders = structMCMC_samples

    if np.any(orders < 0):
        posterior = (orders >= 0).astype(np.int_)
    else:
        posterior = orders

    results = expected_edges(dist=posterior)
    mean, ci = get_mean_and_ci(*results, num_samples=posterior.shape[0], alpha=0.95)
    print(f'E-# Edges: {mean:.2f} ± {ci:.2f}')

    results = expected_shd(dist=posterior, g=gt_adjacency)
    mean, ci = get_mean_and_ci(*results, num_samples=posterior.shape[0], alpha=0.95)
    print(f'E-SHD: {mean:.2f} ± {ci:.2f}')

    metrics = threshold_metrics(dist=posterior, g=gt_adjacency)
    print(f'AUROC: {metrics["roc_auc"]:.3f}')
