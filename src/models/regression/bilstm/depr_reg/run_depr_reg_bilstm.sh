#!/bin/bash
#SBATCH --job-name b0_depr # Name for your job
#SBATCH --ntasks 4              # Number of (cpu) tasks
#SBATCH --time  2800         # Runtime in minutes.
#SBATCH --mem 12000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --gres gpu:gtx1080:1            # Reserve 1 GPU for usage (titartx, gtx1080)
#SBATCH --chdir /nfs_home/nallapar/rb-prof/src/models/regression/bilstm/depr_reg

# RUN BENCHMARK
source /nfs_home/nallapar/rb-prof/bio_embeds/bin/activate
python bilstm_reg.py
