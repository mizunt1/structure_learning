#!/bin/bash
#SBATCH --job-name=mh
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19

module load anaconda/3
conda activate dibs_env
python main.py --name mh_arxiv2_n5 --seed ${SLURM_ARRAY_TASK_ID} mcmc --method mh 
