#!/bin/bash
#SBATCH --job-name transformer_codon # Name for your job
#SBATCH --ntasks 1              # Number of (cpu) tasks
#SBATCH --time  1440         # Runtime in minutes.
#SBATCH --mem 12000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --mail-user vamsi.nallapareddy@epfl.ch     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --gres gpu:gtx1080:1            # Reserve 1 GPU for usage
#SBATCH --chdir /nfs_home/nallapar/rb-prof/src/models

# RUN BENCHMARK
export CUDA_VISIBLE_DEVICES=0
/nfs_home/nallapar/dandl/ml_env/bin/python transformer_reg_fullFT.py
