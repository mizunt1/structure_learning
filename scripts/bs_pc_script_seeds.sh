#!/bin/bash
#SBATCH --job-name=pc
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-14%10
export WANDB_DIR=$SCRATCH/aistats/jobs/$SLURM_ARRAY_JOB_ID
mkdir -p $WANDB_DIR

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
module load singularity/3.7.1

singularity exec bs_env.simg python3 main.py --name 50_nodes --num_variables 50 --num_edges 50 --seed ${seedarr[$SEED_IDX]} bs --method pc
