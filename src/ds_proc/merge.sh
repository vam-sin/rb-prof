#!/bin/bash
#SBATCH --job-name ds_proc # Name for your job
#SBATCH --ntasks 4              # Number of (cpu) tasks
#SBATCH --time  240           # Runtime in minutes.
#SBATCH --mem 12000             # Reserve x GB RAM for the job
#SBATCH --partition cpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --output myjob-%j.txt
#SBATCH --chdir /nfs_home/nallapar/rb-prof/src/ds_proc

# RUN BENCHMARK
/nfs_home/nallapar/dandl/ml_env/bin/python merge_ds.py
