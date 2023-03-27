#!/bin/sh
#SBATCH --job-name=BCD                                                                                                                                                                                                                                                                     
#SBATCH --ntasks=1                                                                                                                                                                                                                                                                         
#SBATCH --gres=gpu:1                                                                                                                                                                                                                                                                       
#SBATCH --time=2:00:00                                                                                                                                                                                                                                                                
#SBATCH --mem=32G  

module load anaconda/3
module list cudatoolkit/11.1

conda activate baseline_bcd_env
python main.py --seed 1 bcd --num_steps 10000