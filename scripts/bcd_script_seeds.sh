#!/bin/bash
#SBATCH --job-name=BCD
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --array=0-19%10
#SBATCH --begin=now+1hour
export WANDB_DIR=$SCRATCH/aistats/jobs/$SLURM_ARRAY_JOB_ID
mkdir -p $WANDB_DIR

module load anaconda/3
module load cudatoolkit/11.1

conda activate baseline_bcd_env
python main.py  --seed ${SLURM_ARRAY_TASK_ID} --num_variables 50 --num_edges 50 --name 50_nodes --graph erdos_renyi_lingauss bcd --num_steps 50000
