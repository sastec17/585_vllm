#!/bin/bash
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node (i.e., number of CPUs)
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --time=02:00:00           # Request 1 hours of runtime (just in case)
#SBATCH --partition=gpu          # Request the GPU partition
#SBATCH --mem=10GB

source ~/.bash_profile
echo "Ensure in correct conda env"

conda env list
echo $CONDA_PREFIX
conda deactivate
echo $CONDA_PREFIX
conda activate bert_env
conda env list
echo $CONDA_PREFIX

echo "verify corrct cuda version selected"
nvcc --version

# run 