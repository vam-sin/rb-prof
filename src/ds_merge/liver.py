'''
this script makes a codon sequence which is in line with the one used in the original liver dataset (standard)
'''

# libraries
import itertools
import pandas as pd 
import numpy as np 
import pickle as pkl 
from os import listdir
from os.path import isfile, join
import os
import csv 

ribo_data_dirpath = '/nfs_home/craigher/scratch/translation_proj/data/liver'

output_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/liver_06_sequence'

with open(os.path.join(ribo_data_dirpath, "counts.csv"), "r") as read_obj:
    counts = list(csv.reader(read_obj))
with open(os.path.join(ribo_data_dirpath, "sequences.csv"), "r") as read_obj:
    sequences = list(csv.reader(read_obj))

# perc_annot list
perc_annot = []
for i in range(len(counts)):
    num_nonzero = np.count_nonzero(np.array(counts[i]).astype(np.float))
    perc_annot.append(num_nonzero / len(counts[i]))

thresh = 0.6
# remove sequences with less than thresh% annotation from the counts and sequences files
counts = [counts[i] for i in range(len(counts)) if perc_annot[i] >= thresh]
sequences = [sequences[i] for i in range(len(sequences)) if perc_annot[i] >= thresh]

print(sequences[0], counts[0])

# for loop to convert sequences to standard format
for i in range(len(sequences)):
    print(i, len(sequences))
    data_dict = {}
    standard_seq = [int(x) for x in sequences[i]]
    data_dict['sequence_standard'] = standard_seq

    # output labels
    data_dict['y'] = [float(x) for x in counts[i]]

    # save the new file
    with open(os.path.join(output_path, 'liver-' + str(i) + '_CTRL_.pkl'), 'wb') as f:
        pkl.dump(data_dict, f)
        f.close()