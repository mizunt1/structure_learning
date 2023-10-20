# Bayesian structure learning benchmarks

This is a repository that includes several Bayesian structure learning methods, data generation from linear Gaussian DAGs, and several metrics to compare their efficacy.

This codebase was developed by [Jithendaraa Subramanian](https://jithendaraa.github.io/) and [Mizu Nishikawa-Toomey](https://mizunt1.github.io/).\
We wanted to create a centralised repository of Bayesian causal structure learning methods and metrics which are easy to access and use. 

This codebase was used to run experiments for the paper [Bayesian learning of Causal Structure and Mechanisms with GFlowNets and Variational Bayes](https://arxiv.org/abs/2211.02763), but we hope it will be part of a ongoing project to collect causal structure learning algorithms. 

If you have a causal structure learning algorithm which you would like to add to the repository, please feel free to reach to build on the codebase. 

Set up instructions for creating an environment that works for dibs, vbg and mcmc methods : 
```
conda create -n envname python=3.10
conda activate envname
conda install jax cuda-nvcc jaxlib==0.4.4=cuda112* cudatoolkit -c conda-forge -c nvidia
pip install -r requirements.txt
pip uninstall -y torch

```
To setup BCD nets:
```
chmod +x scripts/bcd_setup.sh
./scripts/bcd_setup.sh
```
For the bootstrap methods, the environment to run the algorithms was quite involved. We created a singularity instance to reproduce the environment, please reach out if you would like guidance or would like to have a copy of this instance. 

To simulate data from a DAG, and to run inference simply run: 
```
python main.py <inference model name>
```
See the scripts directory for some slurm scripts on how methods were run on slurm at Mila. 

Structure learning methods currently include:
- [VBG](https://arxiv.org/abs/2211.02763) 
- [JSP](https://arxiv.org/abs/2305.19366)
- [DIBS](https://arxiv.org/abs/2105.11839)
- [BCD](https://arxiv.org/abs/2112.02761)
- MCMC (Metropolis hastings / Gibbs)
- Bootstrap GES and PC
