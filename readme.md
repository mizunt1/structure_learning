# Structure learning benchmarks

This is a repository that includes several Bayesian structure learning methods, data generation from linear Gaussian DAGs, and several metrics to compare their efficacy. 


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
