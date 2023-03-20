import dibs
from dibs.inference import JointDiBS
from dibs.target import make_linear_gaussian_model
import jax.random as random

class Model:
    def __init__(self):
        self.subk = None
        self.num_samples_posterior = None
        self.dibs = None

    def train(self, data, num_samples_posterior,
              seed, model_obs_noise, args):
        self.key = random.PRNGKey(seed) 
        self.steps = args.steps
        self.num_samples_posterior = num_samples_posterior
        self.key, self.subk = random.split(self.key)
        self.num_variables  = data.shape[1]
        _, model = make_linear_gaussian_model(key=self.subk, n_vars=self.num_variables, obs_noise=model_obs_noise)
        # sample 10 DAG and parameter particles from the joint posterior
        self.dibs = JointDiBS(x=data.to_numpy(), interv_mask=None, inference_model=model)

    def sample(self):
        gs, thetas = self.dibs.sample(key=self.subk, n_particles=self.num_samples_posterior, steps=self.steps)
        return gs, thetas
