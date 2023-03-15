from gflownet_sl.scores.base import BaseScore, LocalScore
from gflownet_sl.scores.pgmpy_bge_score import BGeScore as BGeScore_


class BGeScore(BaseScore):
    def __init__(
            self,
            data,
            mean_obs=None,
            alpha_mu=1.,
            alpha_w=None,
            prior='uniform',
            n_edges_per_node=1,
            beta=0.1
        ):
        super().__init__(data)
        self.local_scorer = BGeScore_(
            data,
            mean_obs=mean_obs,
            alpha_mu=alpha_mu,
            alpha_w=alpha_w,
            prior=prior,
            beta=beta,
            n_edges_per_node=n_edges_per_node,
            use_variable_names=False
        )
        self.mean_obs = self.local_scorer.mean_obs
        self.alpha_mu = self.local_scorer.alpha_mu
        self.alpha_w = self.local_scorer.alpha_w
        self.prior = self.local_scorer.prior
        self.n_edges_per_node = self.local_scorer.n_edges_per_node
        self.beta = self.local_scorer.beta

    def get_local_scores(self, target, indices, indices_after=None):
        all_indices = indices if (indices_after is None) else indices_after
        local_score_after = self.local_score(target, all_indices)
        if indices_after is not None:
            local_score_before = self.local_score(target, indices)
        else:
            local_score_before = None
        return (local_score_before, local_score_after)

    def local_score(self, target, indices):
        return LocalScore(
            key=(target, tuple(indices)),
            score=self.local_scorer.local_score(target, indices)
        )
