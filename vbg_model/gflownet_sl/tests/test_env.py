import pytest

from pgmpy.estimators import BDeuScore as PgmpyBDeuScore
from pgmpy.utils import get_example_model
from numpy.random import default_rng

from gflownet_sl.env import GFlowNetDAGEnv
from gflownet_sl.scores import BDeuScore
from gflownet_sl.utils.sampling import sample_from_discrete
from gflownet_sl.utils.policy import get_random_actions


def test_local_scores():
    # Sample data from Sachs model
    model = get_example_model('sachs')
    samples = sample_from_discrete(model, num_samples=100, seed=1357)

    # Sample random trajectories to fill in the cache
    rng = default_rng(0)
    env = GFlowNetDAGEnv(
        num_envs=32,
        scorer=BDeuScore(data=samples, equivalent_sample_size=10)
    )
    observations = env.reset()
    for _ in range(10):
        # Get random valid action (based on the mask)
        actions = get_random_actions(observations['mask'], rng, weight=20)
        observations, rewards, dones, _ = env.step(actions)

    # Verify the local scores match the ones from pgmpy
    estimator = PgmpyBDeuScore(data=samples, equivalent_sample_size=10)
    columns = list(samples.columns)
    for ((variable, parents), value) in env.local_scores.items():
        variable = columns[variable]
        parents = tuple(columns[parent] for parent in parents)

        local_score = estimator.local_score(variable, parents)
        assert local_score == value


def test_local_scores_no_multiprocessing():
    # Sample data from Sachs model
    model = get_example_model('sachs')
    samples = sample_from_discrete(model, num_samples=1000, seed=1357)

    # Sample random trajectories to fill in the cache
    rng = default_rng(0)
    env = GFlowNetDAGEnv(
        num_envs=32,
        scorer=BDeuScore(data=samples, equivalent_sample_size=10),
        num_workers=0
    )
    observations = env.reset()
    for _ in range(10):
        # Get random valid action (based on the mask)
        actions = get_random_actions(observations['mask'], rng, weight=20)
        observations, rewards, dones, _ = env.step(actions)

    # Verify the local scores match the ones from pgmpy
    estimator = PgmpyBDeuScore(data=samples, equivalent_sample_size=10)
    columns = list(samples.columns)
    for ((variable, parents), value) in env.local_scores.items():
        variable = columns[variable]
        parents = tuple(columns[parent] for parent in parents)

        local_score = estimator.local_score(variable, parents)
        assert local_score == value
