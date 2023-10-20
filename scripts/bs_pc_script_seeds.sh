#!/bin/bash
#SBATCH --job-name=pc
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:10:00
#SBATCH --mem=10Gb
#SBATCH --array=2-4

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64


module load singularity/3.7.1

singularity exec bs_env.simg python3 main.py --graph sachs --name sachs_20 --seed ${SLURM_ARRAY_TASK_ID} bs --method pc
