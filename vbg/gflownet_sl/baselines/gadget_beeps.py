import sumu
import numpy as np


class BaselineGadget(sumu.Gadget):
    """
    Wrapper for Gadget, from sumu package

    Parameters
    ----------
    data: numpy.array or str
        array with data, or path to .csv (space-separated)
    discrete: boolean (default: False)
        False if data is continuous
    """
    def __init__(self, data, discrete=False):
        data = sumu.Data(data, discrete=discrete)
        params = {"data": data,
                  "scoref": "bge" if not discrete else "bdeu",  # Or "bdeu" for discrete data.
                  "ess": 10,  # If using BDeu.
                  "max_id": -1,
                  "K": min(data.n - 1, 16),  # candidate parents per node
                  "d": min(data.n - 1, 3),  # max size of parent sets that are not subsets of the candidate parents
                  "cp_algo": "greedy-lite",
                  "mc3_chains": 16,
                  "burn_in": 10000,
                  "iterations": 10000,
                  "thinning": 10}
        super(BaselineGadget, self).__init__(**params)


def beeps_from_numpy(dags, data):
    """
    Estimate causal effect from data and sampled DAGs

    Parameters
    ----------
    dags: list of X
        size of sample to be generated

    data: numpy.array or str
        array with data, or path to .csv (space-separated)

    Returns
    -------
    causal_effects: numpy.array
        shape=(len(dags), n*n - n)
    """
    data = sumu.Data(data, discrete=False)
    causal_effects = sumu.beeps(dags, data)
    return causal_effects


if __name__ == "__main__":
    data = np.array([[3.14, -2.041, -0.13],
                     [6.12, -4.083, -0.15],
                     [3.13, -2.039, 0.21],
                     [6.12, -4.082, 0.09],
                     [3.11, -2.040, 0.11]])
    g = BaselineGadget(data, discrete=False)
    dags, scores = g.sample()

    # The following only for continuous data
    dags = [sumu.bnet.family_sequence_to_adj_mat(dag) for dag in dags]
    # causal_effects = sumu.beeps(dags, data)
    causal_effects = beeps_from_numpy(dags, data)  # shape=(len(dags), n*n - n)

    # shape: n*n - n
    # causal effect of X0 on X1,...,Xn, of X1 on X0,X2,...,Xn, ...
    print(causal_effects.mean(axis=0))
