#!/bin/bash
#SBATCH --job-name=BCD
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=8G
#SBATCH --array=0-4
module load anaconda/3
module load cudatoolkit/11.1
conda activate baseline_bcd_env
python main.py  --seed ${SLURM_ARRAY_TASK_ID} --num_variables 50 --num_edges 50 --name 50_nodes_longer --graph erdos_renyi_lingauss bcd --num_steps 10000
