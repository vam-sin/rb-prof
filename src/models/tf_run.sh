#!/bin/bash
#SBATCH --job-name transformer_codon # Name for your job
#SBATCH --ntasks 1              # Number of (cpu) tasks
#SBATCH --time 120             # Runtime in minutes.
#SBATCH --mem 30000             # Reserve 20 GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff
#SBATCH --output out-%j.txt       # Standard out goes to this file
#SBATCH --error err-%j.txt        # Standard err goes to this file
#SBATCH --mail-user vamsi.nallapareddy@epfl.ch     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --gres gpu:titanrtx:1            # Reserve 1 GPU for usage
#SBATCH --chdir /nfs_home/nallapar/rb-prof/src/models


# RUN BENCHMARK
/nfs_home/nallapar/dandl/ml_env/bin/python transformer.py
