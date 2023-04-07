#!/bin/bash
#SBATCH --job-name=BCD
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --array=0-19

module load anaconda/3
module load cudatoolkit/11.1
conda activate baseline_bcd_env
python main.py --seed ${SLURM_ARRAY_TASK_ID} bcd --num_steps 10000
