from dibs.inference import JointDiBS, MarginalDiBS
from dibs.target import make_linear_gaussian_model, make_nonlinear_gaussian_model, make_linear_gaussian_equivalent_model
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
        self.marginal = args.marginal
        self.non_lin = args.non_lin

    def train(self, data, seed):
        key = random.PRNGKey(seed)
        key, subk = random.split(key)
        self.num_variables  = data.shape[1]
        if self.marginal:
            _, model = make_linear_gaussian_equivalent_model(key=subk, n_vars=self.num_variables,
                                                             obs_noise=self.model_obs_noise,
                                                             graph_prior_str=self.prior_str)
        elif self.non_lin:
            _, model = make_nonlinear_gaussian_model(key=subk, n_vars=self.num_variables,
                                                     obs_noise=self.model_obs_noise,
                                                     graph_prior_str=self.prior_str)

        else:
            _, model = make_linear_gaussian_model(key=subk, n_vars=self.num_variables,
                                                  obs_noise=self.model_obs_noise,
                                                  graph_prior_str=self.prior_str)
        # sample 10 DAG and parameter particles from the joint posterior
        if self.marginal:
            self.dibs = MarginalDiBS(x=data.to_numpy(), interv_mask=None, inference_model=model)
        else:
            self.dibs = JointDiBS(x=data.to_numpy(), interv_mask=None,
                              inference_model=model)
        if self.marginal or self.non_lin:
            self.gs = self.dibs.sample(key=subk, n_particles=self.num_samples_posterior,
                                  steps=self.steps)
            self.thetas = self.gs
        else:
            self.gs, self.thetas = self.dibs.sample(key=subk,
                                                    n_particles=self.num_samples_posterior,
                                                    steps=self.steps)
        
    def sample(self):
        if self.marginal:
            if self.plus:
                dist = self.dibs.get_mixture(self.gs)
            else:
                dist = self.dibs.get_empirical(self.gs)
        else:
            if self.plus:
                dist = self.dibs.get_mixture(self.gs, self.thetas)
            else:
                dist = self.dibs.get_empirical(self.gs, self.thetas)
            
        self.gs = dist.g
        if self.marginal:
            self.theta = dist.g
        else:
            self.theta = dist.theta
        is_dag = elwise_acyclic_constr_nograd(self.gs, self.num_variables) == 0
        posterior_graphs = self.gs[is_dag, :, :]
        if self.marginal:
            return posterior_graphs, posterior_graphs, None
        else:
            posterior_thetas = self.thetas[is_dag, :, :]
            return posterior_graphs, posterior_thetas, None
