#!/bin/bash
#SBATCH --job-name=s5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load anaconda/3
conda create -n envname python=3.10
conda activate envname
conda install jax cuda-nvcc jaxlib==0.4.4=cuda112* cudatoolkit -c conda-forge -c nvidia
pip install -r requirements.txt
pip uninstall -y torch
python main.py vbg --num_iterations 10 
