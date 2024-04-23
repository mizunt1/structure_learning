#!/bin/bash
#SBATCH --job-name=n20
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-19

WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
<<<<<<< HEAD
WANDB_ENTITY=$mizunt
export WANDB_DIR=$SCRATCH/aistats/jobs/$SLURM_ARRAY_JOB_ID
mkdir -p $WANDB_DIR
=======
>>>>>>> c6646b34b0f9d70991cdd304c59594a45efa9764

module load anaconda/3
module load cudatoolkit/11.1
conda activate vbg
python main.py  --seed ${SLURM_ARRAY_TASK_ID} --num_variables 20 --num_edges 40 --model_obs_noise 0.1 --data_obs_noise 0.1 --name vbg_20_kl_w0.1 --graph erdos_renyi_lingauss vbg --weight 0.1
