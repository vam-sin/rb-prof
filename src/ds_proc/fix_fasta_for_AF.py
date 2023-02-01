import pandas as pd 
from Bio import SeqIO 
import numpy as np 
import random
import pickle as pkl

records_ds = []
sequences_ds = []

with open('../data/rb_prof_Naef/processed_proper/seq_annot_raw/uniprot_not_matches_withX_withSmallest.fasta') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        records_ds.append(record.description)
        sequences_ds.append(str(record.seq))

f = open('../data/rb_prof_Naef/processed_proper/seq_annot_raw/af_pred.fasta', 'w')

for i in range(len(records_ds)):
    edit_seq = sequences_ds[i]
    if edit_seq[len(edit_seq)-1] == 'X':
        edit_seq = edit_seq[0:len(edit_seq)-1]
    if 'X' in edit_seq:
        edit_seq = edit_seq.replace('X', random.choice(['Y', 'W']))
    print(i, edit_seq)
    f.write('>' + records_ds[i] + '\n')
    f.write(edit_seq + '\n')

f.close()