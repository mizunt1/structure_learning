#!/bin/bash
#SBATCH --job-name=vbg
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-4%

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py --seed ${SLURM_ARRAY_TASK_ID} --num_variables 50 --num_edges 50 --model_obs_noise 0.1 --data_obs_noise 0.1 --name 50_nodes_longer --graph erdos_renyi_lingauss vbg --num_vb_updates 1500 --num_iterations 8
