#!/bin/bash
#SBATCH --job-name=causica
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-4
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load python/3.9
source causica_venv/bin/activate
python main.py --graph sachs --seed ${SLURM_ARRAY_TASK_ID} --name sachs_5 causica
