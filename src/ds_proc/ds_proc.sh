#!/bin/bash
#SBATCH --job-name ds_proc # Name for your job
#SBATCH --ntasks 1              # Number of (cpu) tasks
#SBATCH --time  120           # Runtime in minutes.
#SBATCH --mem 12000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --output myjob-%j.txt
#SBATCH --mail-user vamsi.nallapareddy@epfl.ch     # this is the email you wish to be notified at
#SBATCH --mail-type ALL         # ALL will alert you of job beginning, completion, failure etc
#SBATCH --gres gpu:gtx1080:1            # Reserve 1 GPU for usage
#SBATCH --chdir /nfs_home/nallapar/rb-prof/src/ds_proc

# RUN BENCHMARK
/nfs_home/nallapar/dandl/ml_env/bin/python dataset_pruning.py
