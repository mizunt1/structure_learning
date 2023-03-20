import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd

from gflownet_sl.baselines.bootstrap.bootstrap import NonparametricDAGBootstrap
from gflownet_sl.baselines.bootstrap.learners import PC, GES
from gflownet_sl.utils.metrics import return_file_paths

jax.config.update('jax_platform_name', 'cpu')

def main(args):
    file_paths = return_file_paths(args.seed, args.results, f'bootstrap_{args.method}')
    data = pd.read_csv(file_paths['data'], index_col=0, header=0)
    data = jnp.asarray(data.to_numpy())

    # Code borrowed from DiBS: dibs/eval/joint_inference.py
    if args.method == 'ges':
        boot = NonparametricDAGBootstrap(
            learner=GES(),
            verbose=args.verbose,
            n_restarts=20,  # Default value in DiBS (see dibs/eval/parser.py)
            no_bootstrap=False  # Default value in DiBS (see dibs/eval/joint_inference.py)
        )
    elif args.method == 'pc':
        boot = NonparametricDAGBootstrap(
            learner=PC(
                ci_test='gaussian',  # Default value in DiBS (see dibs/eval/parser.py)
                ci_alpha=0.05  # Default value in DiBS (see dibs/eval/parser.py)
            ),
            verbose=args.verbose,
            n_restarts=20,  # Default value in DiBS (see dibs/eval/parser.py)
            no_bootstrap=False  # Default value in DiBS (see dibs/eval/joint_inference.py)
        )
    else:
        raise NotImplementedError()

    g_samples = boot.sample_particles(
        key=jax.random.PRNGKey(args.seed),
        n_samples=1000,
        x=data,
        verbose_indication=100
    )

    # MLE parameters
    cov_mat = jnp.matmul(data.T, data) / data.shape[0]
    mle_kwargs = {
        'type': 'lingauss',
        'cov_mat': cov_mat,
        'graphs': g_samples,
    }
    theta_samples = boot.learner.get_mle_params(mle_kwargs)

    if not args.dryrun:
        # Save the graphs and thetas
        with open(file_paths['est_post_g'], 'wb') as f:
            np.save(f, np.asarray(g_samples))

        with open(file_paths['est_post_theta'], 'wb') as f:
            np.save(f, np.asarray(theta_samples))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Bootstrap baseline')
    parser.add_argument('--results', type=str, default='results1',
        choices=['results1', 'results2'], help='experiment type (default: %(default)s)')
    parser.add_argument('--method', type=str, default='ges',
        choices=['ges', 'pc'], help='method (default: %(default)s)')
    parser.add_argument('--obs_noise', type=float, default=1.,
        help='observation noise (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
        help='random seed (default: %(default)s)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dryrun', action='store_true')

    args = parser.parse_args()

    main(args)
