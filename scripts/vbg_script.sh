#!/bin/bash
#SBATCH --job-name=s5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64

module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py --seed 3 --model_obs_noise 0.1 --data_obs_noise 0.1 --name sachs_20 --graph sachs vbg --weight 0.5
