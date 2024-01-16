#!/bin/bash
#SBATCH --job-name=s5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load python/3.9
module load cuda/11.2/cudnn/8.1

python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --upgrade pip

pip install -r requirements.txt
pip uninstall -y torch
pip install optax
pip install "jax[cuda]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python main.py mcmc --method mh

