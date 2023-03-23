import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import jax.numpy as jnp
import jax

from mcmc_bs.mcmc_joint.linear_gaussian import LinearGaussianGaussianJAX
from mcmc_bs.mcmc_joint.joint_structure import MHJointStructureMCMC, GibbsJointStructureMCMC

jax.config.update('jax_platform_name', 'cpu')

# dibs/eval/inference.py

class Model:
    def __init__(self):
        self.rng = None


    def train(self, data, num_samples_posterior,
              seed, model_obs_noise, args):
        self.data = data.to_numpy()
        model = LinearGaussianGaussianJAX(
        obs_noise=model_obs_noise,
        mean_edge=0.,
        sig_edge=1.,
        verbose=False
        )
        self.mcmc_run_params = None
        self.mcmc = None
        self.method = args.method
        self.n_vars = self.data.shape[1]
        self.seed = seed
        self.num_samples_posterior = num_samples_posterior
        def ig_log_joint_target(g_mat, theta):
            no_interv_targets = jnp.zeros(self.n_vars, dtype=jnp.bool_)
            return (model.log_prob_parameters(theta=theta, w=g_mat)
                    +model.log_likelihood(theta=theta, w=g_mat, data=self.data, interv_targets=no_interv_targets))

        # Default value in DiBS (see config/baselines.p)y
        theta_prop_sig = 0.001 if (args.method == 'mh') else 0.05
        
        mcmc_init_params = {
            'n_vars': self.n_vars,
            'only_non_covered': False,  # Default value in DiBS (see dibs/eval/parser.py)
            'theta_prop_sig': theta_prop_sig,
            'verbose': False,
        }

        self.mcmc_run_params = {
            'key': jax.random.PRNGKey(self.seed),
            'n_samples': self.num_samples_posterior,
            'theta_shape': model.get_theta_shape(n_vars=self.n_vars),
            'log_joint_target': ig_log_joint_target,
            'burnin': 10,  # Default value in DiBS (see config/baselines.py)
            'thinning': 10  # Default value in DiBS (see config/baselines.py)
        }
        if self.method == 'mh':
            self.mcmc = MHJointStructureMCMC(**mcmc_init_params)
        elif self.method == 'gibbs':
            self.mcmc = GibbsJointStructureMCMC(**mcmc_init_params)
        else:
            raise NotImplementedError()

    def sample(self):
        g_samples, theta_samples = self.mcmc.sample(
            **self.mcmc_run_params,
            verbose_indication=True
        )
        return g_samples, theta_samples, None
