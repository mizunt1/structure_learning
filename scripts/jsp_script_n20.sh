#!/bin/bash
#SBATCH --job-name=n20
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19%10
export WANDB_DIR=$SCRATCH/aistats/jobs/$SLURM_ARRAY_JOB_ID
mkdir -p $WANDB_DIR

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64

module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py --seed ${SLURM_ARRAY_TASK_ID} --num_variables 50 --num_edges 50 --model_obs_noise 0.1 --data_obs_noise 0.1 --name 50_nodes jsp --num_iterations 500000
