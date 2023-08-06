import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
import pickle
import jax

from tqdm import trange
from numpy.random import default_rng

from dag_gflownet.dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.dag_gflownet.utils.factories import get_scorer
from dag_gflownet.dag_gflownet.utils.gflownet import posterior_estimate

class Model:
    def __init__(self, num_samples_posterior, model_obs_noise, args):
        self.num_samples_posterior = num_samples_posterior
        self.model_obs_noise = model_obs_noise
        self.args = args
        self.gflownet = None
        self.params = None
        self.env = None

    def train(self, data, seed):
        key = jax.random.PRNGKey(self.args.seed)
        rng = default_rng(self.args.seed)
        key, subkey = jax.random.split(key)

        # Create the environment
        scorer = get_scorer(self.args, data, rng=rng)
        self.env = GFlowNetDAGEnv(
            num_envs=self.args.num_envs,
            scorer=scorer,
            num_workers=self.args.num_workers,
            context=self.args.mp_context
        )

        # Create the replay buffer
        replay = ReplayBuffer(
            self.args.replay_capacity,
            num_variables=self.env.num_variables
        )

        # Create the GFlowNet & initialize parameters
        self.gflownet = DAGGFlowNet(
            delta=self.args.delta,
            update_target_every=self.args.update_target_every
        )
        optimizer = optax.adam(self.args.lr)
        self.params, state = self.gflownet.init(
            subkey,
            optimizer,
            replay.dummy['adjacency'],
            replay.dummy['mask']
        )
        exploration_schedule = jax.jit(optax.linear_schedule(
            init_value=jnp.array(0.),
            end_value=jnp.array(1. - self.args.min_exploration),
            transition_steps=self.args.num_iterations // 2,
            transition_begin=self.args.prefill,
        ))

        # Training loop
        indices = None
        observations = self.env.reset()
        with trange(self.args.prefill + self.args.num_iterations, desc='Training') as pbar:
            for iteration in pbar:
                # Sample actions, execute them, and save transitions in the replay buffer
                epsilon = exploration_schedule(iteration)
                actions, key, logs = self.gflownet.act(self.params.online, key, observations, epsilon)
                next_observations, delta_scores, dones, _ = self.env.step(np.asarray(actions))
                indices = replay.add(
                    observations,
                    actions,
                    logs['is_exploration'],
                    next_observations,
                    delta_scores,
                    dones,
                    prev_indices=indices
                )
                observations = next_observations

                if iteration >= self.args.prefill:
                    # Update the parameters of the GFlowNet
                    samples = replay.sample(batch_size=self.args.batch_size, rng=rng)
                    params, state, logs = self.gflownet.step(self.params, state, samples)

                    pbar.set_postfix(loss=f"{logs['loss']:.2f}", epsilon=f"{epsilon:.2f}")

        # Evaluate the posterior estimate
        posterior, _ = posterior_estimate(
            self.gflownet,
            params.online,
            self.env,
            key,
            num_samples=self.args.num_samples_posterior,
            desc='Sampling from posterior'
        )

    def sample(self, seed):
        key = jax.random.PRNGKey(seed)
        posterior, _ = posterior_estimate(
            self.gflownet,
            self.params.online,
            self.env,
            key,
            num_samples=self.args.num_samples_posterior,
            desc='Sampling from posterior'
        )

        return posterior, posterior, posterior
