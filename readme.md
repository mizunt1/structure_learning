# Structure learning benchmarks

This is a repository that includes several Bayesian structure learning methods, data generation from linear Gaussian DAGs, and several metrics to compare their efficacy. 

To get set up: 
```
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt

```
For Bootstrap methods, an alternative setup method is required:
```
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install pip install "jax[cuda]==0.3.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements_bs.txt

```
To simulate data from a DAG, and to run inference simply run: 
```
python main.py <inference model name>
```

Structure learning methods currently include:
- [VBG](https://arxiv.org/abs/2211.02763) 
- [DIBS](https://arxiv.org/abs/2105.11839)
- MCMC (Metropolis hastings / Gibbs)
- BS (WIP)
- BCD (WIP)
