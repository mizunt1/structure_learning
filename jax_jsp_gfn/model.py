import jax.numpy as jnp
import numpy as np
import optax
import torch
import jax
import wandb
from pathlib import Path
from tqdm import trange
from numpy.random import default_rng
from jax_jsp_gfn.dag_gflownet.env import GFlowNetDAGEnv
from jax_jsp_gfn.dag_gflownet.gflownet import DAGGFlowNet
from jax_jsp_gfn.dag_gflownet.utils.replay_buffer import ReplayBuffer
from jax_jsp_gfn.dag_gflownet.utils.factories import get_model, get_model_prior
from jax_jsp_gfn.dag_gflownet.utils.gflownet import posterior_estimate
from jax_jsp_gfn.dag_gflownet.utils.jraph_utils import to_graphs_tuple
from jax_jsp_gfn.dag_gflownet.utils.data import load_artifact_continuous

class Model:
    def __init__(self, num_samples_posterior, model_obs_noise, args):
        self.num_samples_posterior = num_samples_posterior
        self.max_epoch = args.num_steps
        self.model_obs_noise = model_obs_noise
        self.args = args
        self.posterior= None
        self.params = None
        self.vardist = None
        self.train_jnp = None

    def train(self, data, seed):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rng = default_rng(seed)
        rng_2 = default_rng(seed + 1000)

        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        
        self.train_jnp = jax.tree_util.tree_map(jnp.asarray, data)

        # Create the environment
        self.env = GFlowNetDAGEnv(
            num_envs=self.args.num_envs,
            num_variables=data.shape[1],
            max_parents=self.args.max_parents
        )

        # Create the replay buffer
        replay = ReplayBuffer(
            self.args.replay_capacity,
            num_variables=self.env.num_variables,
        )

        # Create the model
        # structure learning codebase takes in variance but jsp takes in
        # standard deviation for the model
        obs_scale = np.sqrt(self.model_obs_noise)
        prior_graph = get_model_prior(self.args.prior, 'uniform', self.args)
        print('prior for jsp is uniform only!')
        model = get_model(self.args.model_type, prior_graph, self.train_jnp, obs_scale)

        # Create the GFlowNet & initialize parameters
        self.gflownet = DAGGFlowNet(
            model=model,
            delta=self.args.delta,
            num_samples=self.args.params_num_samples,
            update_target_every=self.args.update_target_every,
            dataset_size=data.shape[0],
            batch_size=self.args.batch_size_data,
        )

        optimizer = optax.adam(self.args.lr)
        self.params, state = self.gflownet.init(
            subkey,
            optimizer,
            replay.dummy['graph'],
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
        normalization = jnp.array(data.shape[0])

        with trange(self.args.prefill + self.args.num_iterations, desc='Training') as pbar:
            for iteration in pbar:
                # Sample actions, execute them, and save transitions in the replay buffer
                epsilon = exploration_schedule(iteration)
                observations['graph'] = to_graphs_tuple(observations['adjacency'])
                actions, key, logs = self.gflownet.act(self.params.online, key, observations, epsilon, normalization)
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
                    self.params, state, logs = self.gflownet.step(self.params, state, samples, self.train_jnp, normalization)

                    pbar.set_postfix(loss=f"{logs['loss']:.2f}")

    def sample(self, seed):
        key = jax.random.PRNGKey(seed)
        posterior, logs = posterior_estimate(
            self.gflownet,
            self.params.online,
            self.env,
            key,
            self.train_jnp,
            num_samples=self.num_samples_posterior,
            desc='Sampling from posterior'
        )
        return posterior, logs['thetas'].squeeze(1).transpose(0, 2, 1), None

