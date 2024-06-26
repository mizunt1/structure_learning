#!/bin/bash
#SBATCH --job-name=ges
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=1-20%10
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
export WANDB_DIR=$SCRATCH/aistats/jobs/$SLURM_ARRAY_JOB_ID
mkdir -p $WANDB_DIR
seedarr=(4 5 6 7 8 9 10 11 12 13 15 16 17 18 19)
N_SEEDS=${#seedarr[@]}
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS))
module load singularity/3.7.1


singularity exec bs_env.simg python3 main.py --name 50_nodes --num_variables 50 --num_edges 50 --seed ${seedarr[$SEED_IDX]} bs --method ges
