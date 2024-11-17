#!/bin/bash

#SBATCH --job-name=try_vllm
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:05:00          
#SBATCH --output=%x_%j.out            # Save output to job_name_jobID.out

export LD_LIBRARY_PATH=/sw/pkgs/arc/cuda/11.6.2/lib64:$LD_LIBRARY_PATH

. /home/gosreya/oops/env/bin/activate
python python_only_dev.py
python simple_vllm_test.py
python python_only_dev.py --quit-dev
