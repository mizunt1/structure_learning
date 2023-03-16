import dibs
from dibs.inference import JointDiBS
from dibs.target import make_linear_gaussian_model
import jax.random as random
key = random.PRNGKey(0)

class Model:
    def __init__(self):
        self.rng = None
        self.key = None
        self.subkey = None
        self.num_samples_posterior = None

    def train(self, data, rng, key, num_samples_posterior,
              num_variables, seed, model_obs_noise, args):
        self.steps = args.steps
        self.num_samples_posterior
        _, model = make_linear_gaussian_model(key=self.key, n_vars=num_variables, obs_noise=model_obs_noise)
        key, subk = random.split(key)
        # sample 10 DAG and parameter particles from the joint posterior
        dibs = JointDiBS(x=data, interv_mask=None, inference_model=model)
        key, self.subk = random.split(self.key)

    def sample(self):
        gs, thetas = dibs.sample(key=self.subk, n_particles=self.num_samples_posterior, steps=self.steps)
        return gs, thetas
