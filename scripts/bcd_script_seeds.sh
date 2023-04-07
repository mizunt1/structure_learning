#!/bin/bash
#SBATCH --job-name=BCD
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --array=0-3                                                    
module load anaconda/3
module load cudatoolkit/11.1
conda activate baseline_bcd_env
seedarr=(6 5 7)
N_SEEDS=${#seedarr[@]}
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS))

python main.py--seed ${seedarr[$SEED_IDX]} --name bcd_arxiv_n20 --num_variables 20 --num_edges 40 bcd --num_steps 10000
