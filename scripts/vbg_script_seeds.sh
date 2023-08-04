#!/bin/bash
#SBATCH --job-name=vbg
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-4

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py --seed ${SLURM_ARRAY_TASK_ID} --model_obs_noise 0.1 --data_obs_noise 0.1 --name sachs_20 --graph sachs vbg --weight 0.5
