#!/bin/bash
module load anaconda/3
module load cudatoolkit/11.1
conda create -n envname python=3.10
conda activate envname
conda install jax cuda-nvcc jaxlib==0.4.4=cuda112* cudatoolkit -c conda-forge -c nvidia
pip install -r requirements.txt
