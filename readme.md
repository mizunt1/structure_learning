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

To simulate data from a DAG, and to run inference simply run: 
```
python main.py <inference model name>
```

Structure learning methods currently include:
- [VBG](https://arxiv.org/abs/2211.02763) 
- [DIBS](https://arxiv.org/abs/2105.11839)
