#!/bin/bash
#SBATCH --job-name=sachs
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-20%10
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64

module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py --seed ${SLURM_ARRAY_TASK_ID} --model_obs_noise 0.1 --data_obs_noise 0.1 --name vbg_sachs_like_w_0.1 --graph sachs vbg --weight 0.1
