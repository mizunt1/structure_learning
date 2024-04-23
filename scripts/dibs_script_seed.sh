#!/bin/bash
#SBATCH --job-name=dibs
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19%10

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64

module load anaconda/3
conda activate vbg
python main.py --name sachs_20  --graph sachs --seed ${SLURM_ARRAY_TASK_ID} dibs --prior_str er --plus
