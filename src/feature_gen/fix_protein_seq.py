# libraries
import pickle as pkl 
import numpy as np
import random
import pandas as pd 
from Bio import SeqIO

# import data 
with open('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_raw/dataset_ProcFeats_2.pkl', 'rb') as f:
    data = pkl.load(f)

'''
2 is the file that has the proper translated proteins.
'''

keys_list = list(data.keys())

seq = []
desc_uni = []

with open("../../mouse_prots/uniprot_mus_musculus.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq.append(str(record.seq))
        desc_uni.append(record.description.split('|')[1])

uniprot_seq = dict(zip(desc_uni, seq))

with open('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_raw/dataset_ProcFeats_3.pkl', 'wb') as f:
    pkl.dump(dict_seqCounts, f)