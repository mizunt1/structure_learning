#!/bin/bash
#SBATCH --job-name=mh
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-4

module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py --name sachs_20 --graph sachs --seed ${SLURM_ARRAY_TASK_ID} mcmc --method mh 
