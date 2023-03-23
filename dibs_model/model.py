from dibs.inference import JointDiBS
from dibs.target import make_linear_gaussian_model
import jax.random as random

class Model:
    def __init__(self, num_samples_posterior, model_obs_noise, args):
        self.num_samples_posterior = num_samples_posterior
        self.model_obs_noise = model_obs_noise
        self.num_variables = None
        self.steps = args.steps
        self.gs = None
        self.thetas = None

    def train(self, data, seed):
        key = random.PRNGKey(seed)
        key, subk = random.split(key)
        self.num_variables  = data.shape[1]
        _, model = make_linear_gaussian_model(key=subk, n_vars=self.num_variables, obs_noise=self.model_obs_noise)
        # sample 10 DAG and parameter particles from the joint posterior
        self.dibs = JointDiBS(x=data.to_numpy(), interv_mask=None, inference_model=model)
        self.gs, self.thetas = self.dibs.sample(key=subk, n_particles=self.num_samples_posterior, steps=self.steps)

    def sample(self):
        return self.gs, self.thetas, None
