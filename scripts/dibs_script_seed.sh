#!/bin/bash
#SBATCH --job-name=dibs_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-4

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64

module load anaconda/3
conda activate vbg
python main.py --name 50_nodes_longer --num_variables 50 --num_edges 50 --graph erdos_renyi_lingauss --seed ${SLURM_ARRAY_TASK_ID} dibs --prior_str er --steps 3000 
