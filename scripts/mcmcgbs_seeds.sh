#!/bin/bash
#SBATCH --job-name=gibbs
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-20%10
#SBATCH --begin=now+6hours
module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py --name 50_nodes --num_variables 50 --num_edges 50 --seed ${SLURM_ARRAY_TASK_ID} mcmc --method gibbs --burnin 50
