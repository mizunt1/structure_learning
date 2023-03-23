import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import jax.numpy as jnp
import jax

from mcmc_bs.bootstrap.bootstrap import NonparametricDAGBootstrap
from mcmc_bs.bootstrap.learners import PC, GES

jax.config.update('jax_platform_name', 'cpu')

class Model:
    def __init__(self, num_samples_posterior, model_obs_noise, args):
        self.num_samples_posterior = num_samples_posterior
        self.model_obs_noise = model_obs_noise
        self.args = args

    def train(self, data, seed):
        self.data = data.to_numpy()
        self.seed = seed
        self.key = jax.random.PRNGKey(self.seed) 
        if self.args.method == 'ges':
            self.boot = NonparametricDAGBootstrap(
                learner=GES(),
                verbose=False,
                n_restarts=20,  # Default value in DiBS (see dibs/eval/parser.py)
                no_bootstrap=False  # Default value in DiBS (see dibs/eval/joint_inference.py)
            )
        elif self.args.method == 'pc':
            self.boot = NonparametricDAGBootstrap(
                learner=PC(
                    ci_test='gaussian',  # Default value in DiBS (see dibs/eval/parser.py)
                    ci_alpha=0.05  # Default value in DiBS (see dibs/eval/parser.py)
                ),
                verbose=False,
                n_restarts=20,  # Default value in DiBS (see dibs/eval/parser.py)
                no_bootstrap=False  # Default value in DiBS (see dibs/eval/joint_inference.py)
            )
        else:
            raise NotImplementedError()

    def sample(self, seed):
        g_samples = self.boot.sample_particles(
        key=jax.random.PRNGKey(seed),
        n_samples=self.num_samples_posterior,
        x=self.data,
        verbose_indication=100
        )

        # MLE parameters
        cov_mat = jnp.matmul(self.data.T, self.data) / self.data.shape[0]
        mle_kwargs = {
            'type': 'lingauss',
            'cov_mat': cov_mat,
            'graphs': g_samples,
        }
        theta_samples = self.boot.learner.get_mle_params(mle_kwargs)
        return g_samples, theta_samples, None
