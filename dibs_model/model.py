from dibs.inference import JointDiBS
from dibs.target import make_linear_gaussian_model
import jax.random as random

import jax.numpy as jnp
from dibs.graph_utils import elwise_acyclic_constr_nograd

def uniform_prior():
    return jnp.array(0.0)

class Model:
    def __init__(self, num_samples_posterior, model_obs_noise, args):
        self.num_samples_posterior = num_samples_posterior
        self.model_obs_noise = model_obs_noise
        self.num_variables = args.num_variables
        self.steps = args.steps
        self.gs = None
        self.thetas = None
        self.args = args
        self.plus = args.plus
        self.prior_str = args.prior_str
        
    def train(self, data, seed):
        key = random.PRNGKey(seed)
        key, subk = random.split(key)
        self.num_variables  = data.shape[1]
        _, model = make_linear_gaussian_model(key=subk, n_vars=self.num_variables, obs_noise=self.model_obs_noise,
                                              graph_prior_str=self.prior_str)
        # sample 10 DAG and parameter particles from the joint posterior
        self.dibs = JointDiBS(x=data.to_numpy(), interv_mask=None,
                              inference_model=model)
        self.gs, self.thetas = self.dibs.sample(key=subk, n_particles=self.num_samples_posterior, steps=self.steps)
        
    def sample(self):
        if self.plus:
            dist = self.dibs.get_mixture(self.gs, self.thetas)
        else:
            dist = self.dibs.get_empirical(self.gs, self.thetas)
        self.gs = dist.g
        self.theta = dist.theta
        is_dag = elwise_acyclic_constr_nograd(self.gs, self.num_variables) == 0
        posterior_graphs = self.gs[is_dag, :, :]
        posterior_thetas = self.thetas[is_dag, :, :]
        return posterior_graphs, posterior_thetas, None
