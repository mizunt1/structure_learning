#!/bin/sh
module load anaconda/3
module list cudatoolkit/11.1
conda create -n baseline_bcd_env python=3.9
conda activate baseline_bcd_env

pip install "jax[cuda]<=0.2.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "jax[cuda]<=0.2.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install cdt==0.5.23 optax==0.0.9 chex==0.0.8 tensorflow_probability==0.13.0 tqdm wandb
pip install numpy==1.20.0 pandas==1.3.1 cython==0.29.24 ott-jax==0.1.14 matplotlib fuzzywuzzy==0.18.0 sumu==0.1.2 python-levenshtein==0.12.2 dm-haiku==0.0.4
pip install termcolor seaborn pgmpy
pip uninstall -y torch

cd bcd_nets/
header_path=`python get_header_path.py`
echo $header_path
cd c_modules/
cython -3 mine.pyx
g++ -I${header_path} -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -o mine.so mine.c
