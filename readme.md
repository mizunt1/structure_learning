# Bayesian structure learning benchmarks

This is a repository that includes several Bayesian structure learning methods, data generation from linear Gaussian DAGs, and several metrics to compare their efficacy.
Primarily to compare existing Bayesian causal structure learning algorithms to Variational Bayes GFlowNet.


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


Structure learning methods currently include:
- [VBG]
- [JSP](https://arxiv.org/abs/2305.19366)
- [DIBS](https://arxiv.org/abs/2105.11839)
- [BCD](https://arxiv.org/abs/2112.02761)
- MCMC (Metropolis hastings / Gibbs)
- Bootstrap GES and PC
