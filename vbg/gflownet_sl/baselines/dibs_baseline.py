import numpy as np
import jax.numpy as jnp

from dibs.inference import MarginalDiBS
from dibs.models import UniformDAGDistributionRejection, BGe


class DiBS:
    def __init__(self, data, **kwargs):
        self.data = np.asarray(data)  # TODO: Assumes continuous data
        self.num_variables = self.data.shape[1]

        graph_dist = UniformDAGDistributionRejection(self.num_variables)  # Uniform prior
        inference_model = BGe(graph_dist=graph_dist, **kwargs)
        self._dibs = MarginalDiBS(x=self.data, inference_model=inference_model)

    def sample(self, key, num_samples, **kwargs):
        return self._dibs.sample(key=key, n_particles=num_samples, **kwargs)
