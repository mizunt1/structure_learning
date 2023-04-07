#!/bin/bash
#SBATCH --job-name=ges
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt

module load singularity/3.7.1

singularity exec --nv bs.simg python3 main.py --name ges_arxiv2_n5 --seed ${SLURM_ARRAY_TASK_ID} bs --method ges
