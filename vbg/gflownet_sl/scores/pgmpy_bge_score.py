import math
import numpy as np

from scipy.special import gammaln
from pgmpy.estimators import StructureScore


def logdet(array):
    _, logdet = np.linalg.slogdet(array)
    return logdet


class BGeScore(StructureScore):
    def __init__(
            self,
            data,
            mean_obs=None,
            alpha_mu=1.,
            alpha_w=None,
            prior='uniform',
            beta=0.1,
            n_edges_per_node=1,
            use_variable_names=True,
            **kwargs
        ):
        num_variables = len(data.columns)
        if mean_obs is None:
            mean_obs = np.zeros((num_variables,))
        if alpha_w is None:
            alpha_w = num_variables + 2.

        self.data = data
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w
        self.prior = prior
        self.beta = beta
        self.n_edges_per_node = n_edges_per_node
        self.use_variable_names = use_variable_names

        self.column_2_index = dict((name, idx)
            for (idx, name) in enumerate(self.data.columns))
        self.num_variables = num_variables
        self.num_samples = self.data.shape[0]
        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)

        T = self.t * np.eye(self.num_variables)
        data = np.asarray(data)
        data_mean = np.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.R = (T + np.dot(data_centered.T, data_centered)
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * np.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)
        )
        all_parents = np.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1))
            - gammaln(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            # log det(T_JJ)^(..) / det(T_II)^(..) for default T
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * math.log(self.t)
        )

        if self.prior == 'uniform':
            self.log_prior = np.zeros((self.num_variables,))
        elif self.prior == 'fair':
            # log(1 / comb(n - 1, k)) ; eggeling:2019
            self.log_prior = (
                - gammaln(self.num_variables + 1)
                + gammaln(self.num_variables - all_parents + 1)
                + gammaln(all_parents + 1)
            )
        elif self.prior == 'edge':
            # log(beta ** k) ; eggeling:2019
            self.log_prior = all_parents * math.log(beta)
        elif self.prior == 'erdos_renyi':
            # k * log(p) + (n - k - 1) * log(1 - p) ; dibs repo
            num_edges = self.num_variables * self.n_edges_per_node # Default value
            p = num_edges / ((self.num_variables * (self.num_variables - 1)) // 2)
            self.log_prior = (all_parents * math.log(p)
                + (self.num_variables - all_parents - 1) * math.log1p(-p))
        else:
            raise NotImplementedError('Prior must be "uniform", "edge", "fair", or "erdos_renyi".')

    def local_score(self, variable, parents):
        if self.use_variable_names:
            target = self.column_2_index[variable]
            indices = [self.column_2_index[parent] for parent in parents]
        else:
            target, indices = variable, parents
        num_parents = len(indices)

        if indices:
            variables = [target] + list(indices)

            # log det(R_II)^(..) / det(R_JJ)^(..)
            log_term_r = (
                0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents)
                * logdet(self.R[np.ix_(indices, indices)])
                - 0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents + 1)
                * logdet(self.R[np.ix_(variables, variables)])
            )
        else:
            # log det(R)^(...)
            log_term_r = (-0.5 * (self.num_samples + self.alpha_w - self.num_variables + 1)
                * np.log(np.abs(self.R[target, target])))

        return self.log_prior[num_parents] + self.log_gamma_term[num_parents] + log_term_r
