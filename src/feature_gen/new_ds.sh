#!/bin/bash
#SBATCH --job-name new_ds # Name for your job
#SBATCH --ntasks 4              # Number of (cpu) tasks
#SBATCH --time  2880           # Runtime in minutes.
#SBATCH --mem 12000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --gres gpu:gtx1080:1            # Reserve 1 GPU for usage
#SBATCH --chdir /nfs_home/nallapar/rb-prof/src/feature_gen

# RUN BENCHMARK
/nfs_home/nallapar/dandl/ml_env/bin/python DNABERT_ft_gen.py
