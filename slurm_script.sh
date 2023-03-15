#!/bin/bash
#SBATCH --job-name=s5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load python/3.7
module load cuda/11.1/cudnn/8.0

python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --upgrade pip
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt

python main.py vbg 
