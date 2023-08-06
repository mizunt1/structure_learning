#!/bin/bash
#SBATCH --job-name=BCD
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
module load anaconda/3
module load cudatoolkit/11.1
conda activate baseline_bcd_env
python main.py --graph sachs --num_edges 11 --seed 0 --name sachs_20 bcd --num_steps 10000
