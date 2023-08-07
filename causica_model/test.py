import fsspec
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.random import default_rng
import random
import os

from pytorch_lightning.callbacks import TQDMProgressBar
from tensordict import TensorDict
from pgmpy.utils import get_example_model
#from vbg.gflownet_sl.utils.graph_plot import graph_to_matrix_sachs
import causica.distributions as cd

from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.lightning.modules.deci_module import DECIModule
from causica.sem.sem_distribution import SEMDistributionModule
from causica.sem.structural_equation_model import ite
from causica.training.auglag import AugLagLRConfig

warnings.filterwarnings("ignore")
test_run = bool(os.environ.get("TEST_RUN", False))  # used by testing to run the notebook as a script
#
seed = 0
rng = default_rng(seed)
rng_2 = default_rng(seed + 1000)
key = random.PRNGKey(seed)
data = pd.read_csv(
    'data/sachs.data.txt',
    delimiter='\t',
    dtype=np.float_
)
graph = get_example_model('sachs')
#adj = graph_to_matrix_sachs(graph, data.columns)
# Standardize data
data = (data - data.mean()) / data.std()
has_edge_weights = False
test_amount = len(data)//3
data_test = data[0:test_amount]
data = data[test_amount:]
num_nodes = data.shape[1]
node_name_to_idx = {key: i for i, key in enumerate(data.keys())}
constraint_matrix = np.full((num_nodes, num_nodes), np.nan, dtype=np.float32)
pl.seed_everything(seed=1)  # set the random seed

lightning_module = DECIModule(
    noise_dist=cd.ContinuousNoiseDist.GAUSSIAN,
    prior_sparsity_lambda=1.0,
    init_rho=1.0,
    init_alpha=0.0,
    auglag_config=AugLagLRConfig(lr_init_dict={"vardist": 1e-2, "icgnn": 3e-4, "noise_dist": 3e-3}),
)


trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=2000,
    fast_dev_run=test_run,
    callbacks=[TQDMProgressBar(refresh_rate=19)],
    enable_checkpointing=False,
)
trainer.fit(lightning_module, datamodule=data)
