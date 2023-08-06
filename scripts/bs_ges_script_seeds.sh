#!/bin/bash
#SBATCH --job-name=ges
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=1-20%10
#SBATCH --begin=now+16hours
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt

module load singularity/3.7.1

singularity exec bs_env.simg python3 main.py --name sachs_20 --graph sachs --seed ${SLURM_ARRAY_TASK_ID} bs --method ges
