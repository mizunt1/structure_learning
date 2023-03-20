import numpy as np
import os
import wandb
import haiku as hk


def slurm_infos():
    return {
        'slurm/job_id': os.getenv('SLURM_JOB_ID'),
        'slurm/job_user': os.getenv('SLURM_JOB_USER'),
        'slurm/job_partition': os.getenv('SLURM_JOB_PARTITION'),
        'slurm/cpus_per_node': os.getenv('SLURM_JOB_CPUS_PER_NODE'),
        'slurm/num_nodes': os.getenv('SLURM_JOB_NUM_NODES'),
        'slurm/nodelist': os.getenv('SLURM_JOB_NODELIST'),
        'slurm/cluster_name': os.getenv('SLURM_CLUSTER_NAME'),
        'slurm/array_task_id': os.getenv('SLURM_ARRAY_TASK_ID')
    }


def table_from_array(array):
    data = [(value,) + index for (index, value) in np.ndenumerate(array)]
    columns = ['data'] + [f'x_{i}' for i in range(array.ndim)]
    return wandb.Table(data=data, columns=columns)

def table_from_dict(dictionary):
    data = list(dictionary.items())
    return wandb.Table(data=data, columns=['key', 'value'])

def scatter_from_dicts(x, dictx, y, dicty, transform=lambda x: x, title=None):
    data = []
    for key in (dictx.keys() & dicty.keys()):
        data.append([transform(dictx[key]), transform(dicty[key])])
    table = wandb.Table(data=data, columns=[x, y])
    return wandb.plot.scatter(table, x, y, title=title)

def return_ordered_data(dict_est, dict_full, transform=lambda x: x, title=None):
    data_est = []
    data_full = []
    for key in dict_full.keys():
        if key in dict_est.keys():
            data_est.append(transform(dict_est[key]))
            data_full.append(transform(dict_full[key]))
        else:
            data_full.append(transform(dict_full[key]))
            data_est.append(0)
    for key in dict_est.keys():
        if key not in dict_full.keys():
            data_est.append(transform(dict_est[key]))
            data_full.append(0)
    return data_full, data_est


