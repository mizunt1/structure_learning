WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
WANDB_ENTITY=$mizunt
module load python/3.9
module load cuda/11.2/cudnn/8.1
module load singularity/3.7.1

python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install "jax[cuda]==0.3.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade pip

pip install -r requirements.txt

singularity exec --nv gflownet_correct3.simg python3 main.py bs --method pc
