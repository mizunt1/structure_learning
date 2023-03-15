import os
from time import time

import sns
import wandb
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from vbg.main import Model
from data_generation import sample_erdos_renyi_linear_gaussian, sample_from_linear_gaussian
from utils import get_weighted_adjacency, edge_marginal_means

from vbg.gflownet_sl.utils.wandb_utils import slurm_infos, table_from_dict, scatter_from_dicts, return_ordered_data

def main(args):
    wandb.init(
        project='vbg_git',
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    wandb.run.summary.update(slurm_infos())

    rng = default_rng(args.seed)
    rng_2 = default_rng(args.seed + 1000)
    if args.graph == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=args.scale_edges,
            low_edges=args.low_edges,
            obs_noise=args.data_obs_noise,
            rng=rng,
            block_small_theta=True
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples_data,
            rng=rng
        )
        data_test = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples_test,
            rng=rng_2
        )
    data.to_csv(os.path.join(wandb.run.dir, 'data_train.csv'))
    data_test.to_csv(os.path.join(wandb.run.dir, 'data_test.csv'))
    wandb.save('data_test.csv', policy='now')
    wandb.save('data_train.csv', policy='now')
    model_args = {}
    time_start = time()
    model = Model()
    import pdb
    pdb.set_trace()
    model_trained = model.train(data, rng, args.num_samples_posterior, args.num_variables, args.seed, args.model_obs_noise,  args)
    posterior_graphs, posterior_edges = model.sample()
    # save posterior samples
    is_dag = elwise_acyclic_constr_nograd(est_posterior_g, n_vars) == 0
    est_posterior_g = est_posterior_g[is_dag, :, :]
    est_posterior_theta = est_posterior_theta[is_dag, :, :]

    with open(os.path.join(wandb.run.dir, 'posterior_graphs.npy'), 'wb') as f:
        np.save(f, posterior_graphs)
    wandb.save('posterior_graphs.npy', policy='now')
    with open(os.path.join(wandb.run.dir, 'posterior_edges.npy'), 'wb') as f:
        np.save(f, posterior_edges)
    wandb.save('posterior_edges.npy', policy='now')

    time_elapsed = time() - start_time
    time_in_hrs = time_elapsed / (60*60)
    weighted_adj = get_weighted_adjacency(graph)
    binary_adj = weighted_adg > 0
    posterior_edges_mean = np.mean(posterior_edges, axis=0)
    if args.num_variables > 5:
        annot = False

    # mean squared errpr of mean of linear mechanisms parameters compared to real 
    mat = [[(weighted_adj[i,j] - posterior_edges_mean[i,j])**2 for j in range(args.num_variables)] for i in range(args.num_variables)]
    sum_ = sum([sum(mat[i]) for i in range(args.num_variables)])
    mean_squared_error_mean = sum_ / (args.num_variables**2)
    wandb.log({'mse of mean': mean_squared_error_mean})

    # plot edge marginals * means of linear mechanisms parameters 
    edge_marginal_means_ = edge_marginal_means(posterior_edges_mean, posterior_graphs)
    plt.clf()
    edge_mm_plot = sns.heatmap(
        edge_marginal_means_, cmap="Blues", annot=annot, annot_kws={"size": 16})
    wandb.log({'edge marginal means': wandb.Image(edge_mm_plot)})

    # plot edge marginals
    edge_marginals = np.sum(posterior_samples, axis=0)/posterior_samples.shape[0]
    plt.clf()
    edge_m_plot = sns.heatmap(
        edge_marginals, cmap="Blues", annot=annot, annot_kws={"size": 16})
    wandb.log({'edge marginals': wandb.Image(edge_m_plot)})

        # log likelihood of unseen data given model of graph
    log_like = -1*LL(posterior_graphs, posterior_edges, data_test.to_numpy(), sigma=np.sqrt(args.model_obs_noise))
    wandb.run.summary.update({"negative log like": log_like})

    # Compute metrics on the posterior estimate
    gt_adjacency = nx.to_numpy_array(graph, weight=None)

    # Expected SHD
    mean_shd = expected_shd(posterior, gt_adjacency)

    # Expected # Edges
    mean_edges = expected_edges(posterior)

    # Threshold metrics
    thresholds = threshold_metrics(posterior, gt_adjacency)

    wandb.run.summary.update({
        'metrics/shd/mean': mean_shd,
        'metrics/edges/mean': mean_edges,
        'metrics/thresholds': thresholds
    })

    if (args.graph in ['erdos_renyi_lingauss']) and (args.num_variables < 6):
        # Default values set by data generation
        # See `sample_erdos_renyi_linear_gaussian` above
        full_posterior = get_full_posterior(
            data, score='lingauss', verbose=True, prior_mean=0., prior_scale=1., obs_scale=args.model_obs_noise)
        # Save full posterior
        with open(os.path.join(wandb.run.dir, 'posterior_full.npz'), 'wb') as f:
            np.savez(f, log_probas=full_posterior.log_probas,
                **full_posterior.graphs.to_dict(prefix='graphs'),
                **full_posterior.closures.to_dict(prefix='closures'),
                **full_posterior.markov.to_dict(prefix='markov')
            )
        full_edge_log_features = get_edge_log_features(full_posterior)
        full_path_log_features = get_path_log_features(full_posterior)
        full_markov_log_features = get_markov_blanket_log_features(full_posterior)
        wandb.run.summary.update({
            'posterior/fufll/edge': table_from_dict(full_edge_log_features),
            'posterior/full/path': table_from_dict(full_path_log_features),
            'posterior/full/markov_blanket': table_from_dict(full_markov_log_features)
        })
        wandb.log({
            'posterior/scatter/edge': scatter_from_dicts('full', full_edge_log_features,
                'estimate', log_features.edge, transform=np.exp, title='Edge features'),
            'posterior/scatter/path': scatter_from_dicts('full', full_path_log_features,
                'estimate', log_features.path, transform=np.exp, title='Path features'),
            'posterior/scatter/markov_blanket': scatter_from_dicts('full', full_markov_log_features,
                'estimate', log_features.markov_blanket, transform=np.exp, title='Markov blanket features')
        })
        full_edge = list(full_edge_log_features.values())
        est_edge = list(log_features.edge.values())

        full_edge_ordered, est_edge_ordered = return_ordered_data(full_edge_log_features,
                log_features.edge, transform=np.exp)
        edge_mse = sk.metrics.mean_squared_error(est_edge_ordered, full_edge_ordered)
        edge_corr = np.corrcoef(full_edge_ordered, est_edge_ordered)[0][1]
        wandb.log({'edge correlation': edge_corr, 'edge mse': edge_mse})
        
        full_path = list(full_path_log_features.values())
        est_path = list(log_features.path.values())
        full_path_ordered, est_path_ordered = return_ordered_data(full_path_log_features,
                log_features.path, transform=np.exp)
        path_mse = sk.metrics.mean_squared_error(est_path_ordered, full_path_ordered)
        path_corr = np.corrcoef(full_path_ordered, est_path_ordered)[0][1]
        wandb.log({'path correlation': path_corr, 'path mse': path_mse})

        full_markov_ordered, est_markov_ordered = return_ordered_data(full_markov_log_features,
                log_features.markov_blanket, transform=np.exp)

        markov_corr = np.corrcoef(full_markov_ordered, est_markov_ordered)[0][1]
        markov_mse = sk.metrics.mean_squared_error(est_markov_ordered, full_markov_ordered)
        wandb.log({'markov correlation': path_corr,  'markov mse': markov_mse})

        # The posterior estimate returns the order in which the edges have been added
        log_features = get_log_features(posterior, data.columns)
        wandb.run.summary.update({
            'posterior/estimate/edge': table_from_dict(log_features.edge),
            'posterior/estimate/path': table_from_dict(log_features.path),
            'posterior/estimate/markov_blanket': table_from_dict(log_features.markov_blanket)
        })

if __name__ == '__main__':
    from argparse import ArgumentParser
    import json
    parser = ArgumentParser('GFlowNet for Structure Learning')
    subparsers = parser.add_subparsers()
    vbg_parser = subparsers.add_parser("vbg")
    # graph generation
    parser.add_argument('--num_variables', type=int, default=5,
        help='Number of variables (nodes) (default: %(default)s)')
    parser.add_argument('--num_edges', type=int, default=5,
        help='Average number of parents (default: %(default)s)')
    parser.add_argument('--graph', type=str, default='erdos_renyi_lingauss',
                        choices=['erdos_renyi_lingauss'], help='Type of graph (default: %(default)s)')
    parser.add_argument('--data_obs_noise', type=float, default=0.1,
                        help='likelihood variance in data generation')
    parser.add_argument('--scale_edges', type=float, default=2.0,
                        help='upper limit for edge scale')
    parser.add_argument('--low_edges', type=float, default=0.5,
                        help='lower limit for edge scale')
    # data generation
    parser.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')
    parser.add_argument('--num_samples_data', type=int, default=100,
        help='Number of samples (default: %(default)s)')
    parser.add_argument('--num_samples_test', type=int, default=100,
        help='Number of samples (default: %(default)s)')
    # others
    parser.add_argument('--num_samples_posterior', type=int, default=1000,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    parser.add_argument('--model_obs_noise', type=float, default=0.1,
                        help='likelihood variance in approximate posterior')
    
    # VBG args
    vbg_parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size (default: %(default)s)')
    vbg_parser.add_argument('--num_iterations', type=int, default=20,
                            help='Number of iterations (default: %(default)s)')
    vbg_parser.add_argument('--lr', type=float, default=1e-5,
                            help='Learning rate (default: %(default)s)')
    vbg_parser.add_argument('--num_vb_updates', type=int, default=2000,
                           help='number of updates to gflownet per one update of parameters in VB setup')
    vbg_parser.add_argument('--weight', type=float, default=0.1,
                            help='amount of weighting of reward')
    vbg_parser.add_argument('--delta', type=float, default=1.,
                            help='Value of delta for Huber loss (default: %(default)s)')
    vbg_parser.add_argument('--prefill', type=int, default=1000,
                            help='Number of iterations with a random policy to prefill ')
    vbg_parser.add_argument('--num_envs', type=int, default=8,
                            help='Number of parallel environments (default: %(default)s)')
    vbg_parser.add_argument('--update_target_every', type=int, default=1000,
                            help='Frequency of update for the target network (default: %(default)s)')
    vbg_parser.add_argument('--n_step', type=int, default=1,
                            help='Maximum number of subsequences for multistep loss (default: %(default)s)')
    vbg_parser.add_argument('--replay_capacity', type=int, default=100_000,
                            help='Capacity of the replay buffer (default: %(default)s)')
    vbg_parser.add_argument('--replay_prioritized', action='store_true',
                            help='Use Prioritized Experience Replay')
    vbg_parser.add_argument('--min_exploration', type=float, default=0.1,
                            help='Minimum value of epsilon-exploration (default: %(default)s)')
    vbg_parser.add_argument('--start_to_increase_eps', type=float, default=0.5,
                            help='the fraction of training iters to start increasing epsilon')
    vbg_parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of workers (default: %(default)s)')
    vbg_parser.add_argument('--mp_context', type=str, default='spawn',
                            help='Multiprocessing context (default: %(default)s)')
    vbg_parser.add_argument('--use_erdos_prior', default=False,
                            action='store_true',
                            help='whether to use erdos renyi prior over graphs')
    vbg_parser.add_argument('--keep_epsilon_constant', default=False,
                            action='store_true',
                            help='do not increase epsilon over time')
    args = parser.parse_args()
    main(args)
