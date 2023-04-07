#!/bin/bash
#SBATCH --job-name=pc
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt

module load singularity/3.7.1

singularity exec bs_env.simg python3 main.py --seed ${SLURM_ARRAY_TASK_ID} --name pc_arxiv2_n20 --num_variables 20 --num_edges 40 bs --method pc
