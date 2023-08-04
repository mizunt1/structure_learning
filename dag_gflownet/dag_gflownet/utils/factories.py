from numpy.random import default_rng

from dag_gflownet.dag_gflownet.scores import BDeScore, BGeScore, priors
from dag_gflownet.dag_gflownet.utils.data import get_data


def get_prior(name, **kwargs):
    prior = {
        'uniform': priors.UniformPrior,
        'erdos_renyi': priors.ErdosRenyiPrior,
        'edge': priors.EdgePrior,
        'fair': priors.FairPrior
    }
    return prior[name](**kwargs)


def get_scorer(args, data, rng=default_rng()):
    # Get the data
    if args.graph == 'erdos_renyi_lingauss':
        score = 'bge'
    if args.graph == 'sachs':
        score = 'bge'
    # Get the prior
    prior = get_prior(args.prior, **args.prior_kwargs)

    # Get the scorer
    scores = {'bde': BDeScore, 'bge': BGeScore}
    scorer = scores[score](data, prior, **args.scorer_kwargs)

    return scorer
