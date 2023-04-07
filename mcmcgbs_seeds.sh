#!/bin/bash
#SBATCH --job-name=gibbs
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19

module load anaconda/3
conda activate dibs_env
python main.py --seed ${SLURM_ARRAY_TASK_ID} mcmc --method gibbs
