import os
from operator import itemgetter
import warnings
from numpy.random import default_rng
from dataclasses import dataclass

import fsspec
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from causica.training.auglag import AugLagLossCalculator, AugLagLR, AugLagLRConfig
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
from tensordict import TensorDict
from pgmpy.utils import get_example_model
from causica.functional_relationships import ICGNN
from causica.datasets.tensordict_utils import tensordict_from_pandas, tensordict_shapes

import causica.distributions as cd
from causica.lightning.deci_module import DECIModule
from causica.datasets.variable_types import VariableTypeEnum
from causica.graph.dag_constraint import calculate_dagness

from causica.training.auglag import AugLagLRConfig

class Model:
    def __init__(self, num_samples_posterior, model_obs_noise, args):
        self.num_samples_posterior = num_samples_posterior
        self.max_epoch = args['num_steps']
        self.vardist = None

    def train(self, data, seed):
        rng = default_rng(seed)
        rng_2 = default_rng(seed + 1000)
        node_name_to_idx = {key: i for i, key in enumerate(data.keys())}
        constraint_matrix = np.full((num_nodes, num_nodes), np.nan, dtype=np.float32)
        # Training config

        @dataclass(frozen=True)
        class TrainingConfig:
            noise_dist=cd.ContinuousNoiseDist.GAUSSIAN
            batch_size=124
            max_epoch=self.max_epoch
            gumbel_temp=0.25
            averaging_period=10
            prior_sparsity_lambda=5.0
            init_rho=1.0
            init_alpha=0.0

        training_config = TrainingConfig()
        auglag_config = AugLagLRConfig()

        # Cast data to torch tensors

        tensor_dict = tensordict_from_pandas(data)
        tensor_dict = tensor_dict.apply(lambda t: t.to(dtype=torch.float32, device=device))


        # Create loader
        dataloader_train = DataLoader(
            dataset=tensor_dict,
            collate_fn=lambda x: x,
            batch_size=training_config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        prior = cd.GibbsDAGPrior(
            num_nodes=num_nodes, 
            sparsity_lambda=training_config.prior_sparsity_lambda)
        # Define the adjaceny module
        adjacency_dist = cd.ENCOAdjacencyDistributionModule(num_nodes)

        #Define the functional module
        icgnn = ICGNN(
            variables=tensordict_shapes(tensor_dict),
            embedding_size=32,
            out_dim_g=32,
            norm_layer=torch.nn.LayerNorm,
            res_connection=True,
        )

        # Define the noise module
        types_dict = {var_name: VariableTypeEnum.CONTINUOUS for var_name in tensor_dict.keys()}

        noise_submodules = cd.create_noise_modules(
            shapes=tensordict_shapes(tensor_dict), 
            types=types_dict, 
            continuous_noise_dist=training_config.noise_dist
        )

        noise_module = cd.JointNoiseModule(noise_submodules)
        sem_module = cd.SEMDistributionModule(
            adjacency_module=adjacency_dist, 
            functional_relationships=icgnn, 
            noise_module=noise_module)

        sem_module.to(device)

        modules = {
            "icgnn": sem_module.functional_relationships,
            "vardist": sem_module.adjacency_module,
            "noise_dist": sem_module.noise_module,
        }

        parameter_list = [
            {"params": module.parameters(), "lr": auglag_config.lr_init_dict[name], "name": name}
            for name, module in modules.items()
        ]

        # Define the optimizer
        optimizer = torch.optim.Adam(parameter_list)
        scheduler = AugLagLR(config=auglag_config)

        auglag_loss = AugLagLossCalculator(
            init_alpha=training_config.init_alpha, 
            init_rho=training_config.init_rho
        )
        num_samples = len(tensor_dict)

        for epoch in range(training_config.max_epoch):
            for i, batch in enumerate(dataloader_train):
                # Zero the gradients
                optimizer.zero_grad()

                # Get SEM 
                sem_distribution = sem_module()
                sem, *_ = sem_distribution.relaxed_sample(
                    torch.Size([]), 
                    temperature=training_config.gumbel_temp
                )  # soft sample

                # Compute the log probability of data
                batch_log_prob = sem.log_prob(batch).mean()

                # Get the distribution entropy
                sem_distribution_entropy = sem_distribution.entropy()

                # Compute the likelihood of the current graph
                prior_term = prior.log_prob(sem.graph)

                # Compute the objective
                objective = (-sem_distribution_entropy - prior_term) / num_samples - batch_log_prob

                # Compute the DAG-ness term
                constraint = calculate_dagness(sem.graph)

                # Compute the Lagrangian loss
                loss = auglag_loss(objective, constraint / num_samples)

                # Propagate gradients and update
                loss.backward()
                optimizer.step()

                # Update the Auglag parameters
                scheduler.step(
                    optimizer=optimizer,
                    loss=auglag_loss,
                    loss_value=loss.item(),
                    lagrangian_penalty=constraint.item(),
                )

                # Log metrics & plot the matrices
                if epoch % 10 == 0 and i == 0:
                    print(
                        f"epoch:{epoch} loss:{loss.item():.5g} nll:{-batch_log_prob.detach().cpu().numpy():.5g} "
                        f"dagness:{constraint.item():.5f} num_edges:{(sem.graph > 0.0).sum()} "
                        f"alpha:{auglag_loss.alpha:.5g} rho:{auglag_loss.rho:.5g} "
                        f"step:{scheduler.outer_opt_counter}|{scheduler.step_counter} "
                        f"num_lr_updates:{scheduler.num_lr_updates}"
                    )

            self.vardist = adjacency_dist()


    def sample(self):
            samples = self.vardist.sample_n(self.num_samples_posterior).cpu().numpy()
            return samples, samples


if __name__ == "__main__":
    data = pd.read_csv(
        'data/sachs.data.txt',
        delimiter='\t',
        dtype=np.float_
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    graph = get_example_model('sachs')

    #adj = graph_to_matrix_sachs(graph, data.columns)
    # Standardize data
    data = (data - data.mean()) / data.std()
    has_edge_weights = False
    test_amount = len(data)//3
    data_test = data[0:test_amount]
    data = data[test_amount:]
    num_nodes = data.shape[1]
    model = Model(100, 0.1, {'num_steps':100})
    seed = 0
    model.train(data, seed)
    graphs = model.sample()
