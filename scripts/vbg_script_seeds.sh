#!/bin/bash
#SBATCH --job-name=vbg20
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py --seed ${SLURM_ARRAY_TASK_ID} --model_obs_noise 0.1 --data_obs_noise 0.1 --name vbg_arxiv2_w_0.2 vbg --weight 0.2
