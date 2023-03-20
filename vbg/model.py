import jax.numpy as jnp
import numpy as np
import optax

from tqdm import trange
from numpy.random import default_rng
from jax import random
import jax

from collections import namedtuple
from vbg.gflownet_sl.env import GFlowNetDAGEnv
from vbg.gflownet_sl.scores import BGeScore
from vbg.gflownet_sl.gflownet import GFlowNet
from vbg.gflownet_sl.replay_buffer import ReplayBuffer
from vbg.gflownet_sl.utils.jnp_utils import get_random_actions
from vbg.gflownet_sl.utils.gflownet import update_parameters_full
from vbg.gflownet_sl.utils.gflownet import compute_delta_score_lingauss_full
from vbg.gflownet_sl.utils.metrics import posterior_estimate

NormalParameters = namedtuple('NormalParameters', ['mean', 'precision'])
class Model:
    def __init__(self): 
        self.params = None
        self.env_post = None
        self.state_key = None
        self.key = None
        self.num_samples_posterior = None
        self.edge_params = None

    def train(self, data, num_samples_posterior,
              seed, model_obs_noise, args):
        
        self.key = random.PRNGKey(seed)
        self.rng = default_rng(seed)
        self.num_samples_posterior = num_samples_posterior
        num_variables = self.data.shape[1]
        scorer_cls = BGeScore
        env_kwargs = dict()
        scorer_kwargs = {
            'mean_obs': np.zeros(num_variables),
            'alpha_mu': 1.,
            'alpha_w': num_variables + 2.
        }

        env = GFlowNetDAGEnv(
            num_envs=args.num_envs,
            scorer=scorer_cls(data, **scorer_kwargs),
            num_workers=args.num_workers,
            context=args.mp_context,
            vb=True,
            **env_kwargs
        )

        env_post = GFlowNetDAGEnv(
            num_envs=args.num_envs,
            scorer=scorer_cls(data, **scorer_kwargs),
            num_workers=args.num_workers,
            context=args.mp_context,
            vb=True,
            **env_kwargs
        )

        # Create the replay buffer
        replay = ReplayBuffer(
            args.replay_capacity,
            num_variables=env.num_variables,
            n_step=args.n_step,
            prioritized=args.replay_prioritized
        )

        # Create the GFlowNet & initialize parameters
        scheduler = optax.piecewise_constant_schedule(args.lr, {
            # int(0.4 * args.num_iterations): 0.1,
            # int(0.6 * args.num_iterations): 0.05,
            # int(0.8 * args.num_iterations): 0.01
        })
        
        gflownet = GFlowNet(
            optimizer=optax.adam(scheduler),
            delta=args.delta,
            n_step=args.n_step
        )
        params, state = gflownet.init(self.key, replay.dummy_adjacency)

        # Collect data (using random policy) to start filling the replay buffer
        observations = env.reset()
        indices = None
        for _ in trange(args.prefill, desc='Collect data'):
            # Sample random action
            actions, state = get_random_actions(state, observations['mask'])

            # Execute the actions and save the transitions to the replay buffer
            next_observations, rewards, dones, _ = env.step(np.asarray(actions))
            is_exploration = jnp.ones_like(actions)  # All these actions are from exploration step
            indices = replay.add(
                observations,
                actions,
                is_exploration,
                next_observations,
                rewards,
                dones,
                prev_indices=indices
            )
            observations = next_observations

        # Training loop
        tau = jnp.array(1.)  # Temperature for the posterior (should be equal to 1)
        epsilon = jnp.array(0.)
        num_samples = data.shape[0]
        first_run = True
        xtx = jnp.einsum('nk,nl->kl', data.to_numpy(), data.to_numpy())
        prior = NormalParameters(
            mean=jnp.zeros((num_variables,)), precision=jnp.eye((num_variables)))

        with trange(args.num_iterations, desc='Training') as pbar:
            for iteration in pbar:
                losses = np.zeros(args.num_vb_updates)           
                if (iteration + 1) % args.update_target_every == 0:
                    # Update the parameters of the target network
                    gflownet.set_target(params)
                # sample from posterior of graphs without adding to the environment
                if first_run:
                    for _ in range(100):
                        actions, is_exploration, next_state = gflownet.act(params, state, observations, epsilon)
                        next_observations, rewards, dones, _ = env.step(np.asarray(actions))
                        indices = replay.add(
                            observations,
                            actions,
                            is_exploration,
                            next_observations,
                            rewards,
                            dones,
                            prev_indices=indices
                        )
                        observations = next_observations
                        state = next_state
                        samples, subsq_mask = replay.sample(batch_size=args.batch_size, rng=self.rng)
                        params, state, logs = gflownet.step(
                            params,
                            gflownet.target_params,
                            state,
                            samples,
                            subsq_mask,
                            tau
                        )
                    edge_params = prior
                first_run = False
                orders = posterior_estimate(
                    params,
                    env_post,
                    state.key,
                    num_samples=self.num_samples_posterior
                )
                posterior_samples = (orders >= 0).astype(np.int_)
                env_post.reset()
                new_edge_params = update_parameters_full(prior,
                                                         posterior_samples,
                                                         data.to_numpy(),
                                                         model_obs_noise)
                diff_mean = jnp.sum(abs(edge_params.mean - new_edge_params.mean)) / (edge_params.mean.shape[0]**2)
                diff_prec = jnp.sum(abs(edge_params.precision - new_edge_params.precision)) / (edge_params.mean.shape[0]**2)
                edge_params = new_edge_params
                for vb_iters in range(args.num_vb_updates):
                    if vb_iters == 0:
                        state = gflownet.reset_optim(state, params, self.key)
                        epsilon = jnp.array(0.)
                        # only update epsilon if we are half way through training
                    if iteration > (args.num_iterations * args.start_to_increase_eps):
                        if not args.keep_epsilon_constant:
                             epsilon = jnp.minimum(
                                1-args.min_exploration,
                                ((1-args.min_exploration)*2/args.num_vb_updates)*vb_iters)
                        else:
                            epsilon = jnp.array(0.)            
                    # Sample actions, execute them, and save transitions to the buffer
                    actions, is_exploration, next_state = gflownet.act(params, state, observations, epsilon)
                    next_observations, rewards, dones, _ = env.step(np.asarray(actions))
                    indices = replay.add(
                        observations,
                        actions,
                        is_exploration,
                        next_observations,
                        rewards,
                        dones,
                        prev_indices=indices
                    )
                    observations = next_observations
                    state = next_state

                    samples, subsq_mask = replay.sample(batch_size=args.batch_size, rng=self.rng)
                    diff_marg_ll = jax.vmap(
                        compute_delta_score_lingauss_full, in_axes=(0,0,None,None,None,
                                                                    None, None,None))(
                                                                        samples['adjacency'][0],
                                                                        samples['actions'][0],
                                                                        edge_params,
                                                                        prior,
                                                                        xtx,
                                                                        model_obs_noise,
                                                                        args.weight,
                                                                        args.use_erdos_prior)

                    samples['rewards'][0] = diff_marg_ll
                    params, state, logs = gflownet.step(
                        params,
                        gflownet.target_params,
                        state,
                        samples,
                        subsq_mask,
                        tau
                    )
                    replay.update_priorities(samples, logs['error'])
        self.state_key = state.key
        self.edge_params = edge_params
        self.env_post = env_post
        self.params = params

    def sample(self):
        orders = posterior_estimate(
            self.params,
            self.env_post,
            self.state_key,
            num_samples=self.num_samples_posterior
        )
        posterior_graphs = (orders >= 0).astype(np.int_)
        edge_cov = jax.vmap(jnp.linalg.inv, in_axes=-1, out_axes=-1)(self.edge_params.precision)
        posterior_edges = jax.vmap(
            random.multivariate_normal, in_axes=(None, -1, -1, None),  out_axes=(-1))(self.key,
                                                                                      self.edge_params.mean,
                                                                                      edge_cov, (self.num_samples_posterior,))

        return posterior_graphs, posterior_edges

