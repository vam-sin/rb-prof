#!/bin/bash
#SBATCH --job-name TF-Reg # Name for your job
#SBATCH --ntasks 1              # Number of (cpu) tasks
#SBATCH --time  1440         # Runtime in minutes.
#SBATCH --mem 12000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --gres gpu:titanrtx:1            # Reserve 1 GPU for usage
#SBATCH --chdir /nfs_home/nallapar/rb-prof/src/models/regression

# RUN BENCHMARK
/nfs_home/nallapar/dandl/ml_env/bin/python transformer_reg.py
