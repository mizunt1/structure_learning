#!/bin/bash
#SBATCH --job-name=dibs_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load anaconda/3
conda activate dibs_env
python main.py --name dibs_plus_arxiv2_n20 --num_edges 40 --num_variables 20 --seed ${SLURM_ARRAY_TASK_ID} dibs --plus
