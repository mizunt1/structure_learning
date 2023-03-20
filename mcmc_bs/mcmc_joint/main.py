import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
import pickle

from gflownet_sl.baselines.mcmc_joint.linear_gaussian import LinearGaussianGaussianJAX
from gflownet_sl.baselines.mcmc_joint.FCGaussian import FCGaussianJAX
from gflownet_sl.baselines.mcmc_joint.joint_structure import MHJointStructureMCMC, GibbsJointStructureMCMC
from gflownet_sl.utils.metrics import return_file_paths

jax.config.update('jax_platform_name', 'cpu')

def main(args):
    file_paths = return_file_paths(args.seed, args.results, f'mcmc_{args.method}')
    data = pd.read_csv(file_paths['data'], index_col=0, header=0)
    data = jnp.asarray(data.to_numpy())
    n_vars = data.shape[1]

    # Code borrowed from DiBS: dibs/eval/joint_inference.py & dibs/eval/inference.py
    # Model
    if args.non_linear:
        model = FCGaussianJAX(
            obs_noise=args.obs_noise,
            sig_param=1.0,  # Default value in DiBS (see dibs/eval/parser.py)
            dims=[5],  # Default value in DiBS (see dibs/eval/parser.py & dibs/eval/target.py)
            verbose=args.verbose,
            activation='relu',  # Default value in DiBS (see dibs/eval/parser.py)
            bias=True  # Default value in DiBS (see dibs/eval/parser.py)
        )
    else:
        model = LinearGaussianGaussianJAX(
            obs_noise=args.obs_noise,
            mean_edge=0.,
            sig_edge=1.,
            verbose=args.verbose
        )

    # dibs/eval/inference.py
    @jax.jit
    def ig_log_joint_target(g_mat, theta):
        no_interv_targets = jnp.zeros(n_vars, dtype=jnp.bool_)
        return (model.log_prob_parameters(theta=theta, w=g_mat)
            + model.log_likelihood(theta=theta, w=g_mat, data=data, interv_targets=no_interv_targets))

    # Default value in DiBS (see config/baselines.py)
    if args.non_linear:
        theta_prop_sig = 0.0001 if (args.method == 'mh') else 0.005
    else:
        theta_prop_sig = 0.001 if (args.method == 'mh') else 0.05

    mcmc_init_params = {
        'n_vars': n_vars,
        'only_non_covered': False,  # Default value in DiBS (see dibs/eval/parser.py)
        'theta_prop_sig': theta_prop_sig,
        'verbose': args.verbose,
    }

    mcmc_run_params = {
        'key': jax.random.PRNGKey(args.seed),
        'n_samples': 1000,
        'theta_shape': model.get_theta_shape(n_vars=n_vars),
        'log_joint_target': ig_log_joint_target,
        'burnin': 10,  # Default value in DiBS (see config/baselines.py)
        'thinning': 10  # Default value in DiBS (see config/baselines.py)
    }

    if args.method == 'mh':
        mcmc = MHJointStructureMCMC(**mcmc_init_params)
    elif args.method == 'gibbs':
        mcmc = GibbsJointStructureMCMC(**mcmc_init_params)
    else:
        raise NotImplementedError()

    g_samples, theta_samples = mcmc.sample(
        **mcmc_run_params,
        verbose_indication=args.verbose
    )

    if not args.dryrun:
        # Save the graphs and thetas
        with open(file_paths['est_post_g'], 'wb') as f:
            np.save(f, np.asarray(g_samples))

        filename = file_paths['est_post_theta']
        if args.non_linear:
            filename, _ = os.path.splitext(filename)
            theta_samples = jax.tree_util.tree_map(np.asarray, theta_samples)
            with open(f'{filename}.pkl', 'wb') as f:
                pickle.dump(theta_samples, f)
        else:
            with open(filename, 'wb') as f:
                np.save(f, np.asarray(theta_samples))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('MCMC baseline')
    parser.add_argument('--results', type=str, default='results1',
        choices=['results1', 'results2'], help='experiment type (default: %(default)s)')
    parser.add_argument('--method', type=str, default='mh',
        choices=['mh', 'gibbs'], help='method (default: %(default)s)')
    parser.add_argument('--obs_noise', type=float, default=1.,
        help='observation noise (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
        help='random seed (default: %(default)s)')
    parser.add_argument('--non_linear', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dryrun', action='store_true')

    args = parser.parse_args()

    main(args)
