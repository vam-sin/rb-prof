'''
this script makes a codon sequence which is in line with the one used in the liver dataset
'''
# libraries
import itertools
import pandas as pd 
import numpy as np 
import pickle as pkl 
from os import listdir
from os.path import isfile, join

liver_number_to_codon = {idx+1:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G', 'N'], repeat=3))}
liver_codon_to_number = {v: k for k, v in liver_number_to_codon.items()}

ds06_codon_to_number = {
        'ATA':1, 'ATC':2, 'ATT':3, 'ATG':4,
        'ACA':5, 'ACC':6, 'ACG':7, 'ACT':8,
        'AAC':9, 'AAT':10, 'AAA':11, 'AAG':12,
        'AGC':13, 'AGT':14, 'AGA':15, 'AGG':16,                
        'CTA':17, 'CTC':18, 'CTG':19, 'CTT':20,
        'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24,
        'CAC':25, 'CAT':26, 'CAA':27, 'CAG':28,
        'CGA':29, 'CGC':30, 'CGG':31, 'CGT':32,
        'GTA':33, 'GTC':34, 'GTG':35, 'GTT':36,
        'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40,
        'GAC':41, 'GAT':42, 'GAA':43, 'GAG':44,
        'GGA':45, 'GGC':46, 'GGG':47, 'GGT':48,
        'TCA':49, 'TCC':50, 'TCG':51, 'TCT':52,
        'TTC':53, 'TTT':54, 'TTA':55, 'TTG':56,
        'TAC':57, 'TAT':58, 'TAA':59, 'TAG':60,
        'TGC':61, 'TGT':62, 'TGA':63, 'TGG':64, 'NNG': 66, 'NGG': 67, 'NNT': 68,
        'NTG': 69, 'NAC': 70, 'NNC': 71, 'NCC': 72,
        'NGC': 73, 'NCA': 74, 'NGA': 75, 'NNA': 76,
        'NAG': 77, 'NTC': 78, 'NAT': 79, 'NGT': 80,
        'NCG': 81, 'NTT': 82, 'NCT': 83, 'NAA': 84,
        'NTA': 85
    }

ds06_number_to_codon = {v: k for k, v in ds06_codon_to_number.items()}

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
train_files = [mypath + '/' + f for f in onlyfiles]

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_files = [mypath + '/' + f for f in onlyfiles]

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
val_files = [mypath + '/' + f for f in onlyfiles]

# # convert sequence for train
# for i in range(len(train_files)):
#     print(i, train_files[i])
#     tr_id = train_files[i].split('/')[-1].split('_')[0]
#     with open(train_files[i], 'rb') as f:
#         data_dict = pkl.load(f)
#         f.close()
    
#     X_seq_orig = np.asarray(data_dict['sequence'])

#     # convert to liver codon format
#     X_seq_standard = [liver_codon_to_number[ds06_number_to_codon[el]] for el in X_seq_orig]

#     data_dict['sequence_standard'] = X_seq_standard

#     # save the new file
#     with open(train_files[i], 'wb') as f:
#         pkl.dump(data_dict, f)
#         f.close()

# # convert sequence for val
# for i in range(len(val_files)):
#     print(i, val_files[i])
#     tr_id = val_files[i].split('/')[-1].split('_')[0]
#     with open(val_files[i], 'rb') as f:
#         data_dict = pkl.load(f)
#         f.close()
    
#     X_seq_orig = np.asarray(data_dict['sequence'])

#     # convert to liver codon format
#     X_seq_standard = [liver_codon_to_number[ds06_number_to_codon[el]] for el in X_seq_orig]

#     data_dict['sequence_standard'] = X_seq_standard

#     # save the new file
#     with open(val_files[i], 'wb') as f:
#         pkl.dump(data_dict, f)
#         f.close()

# convert sequence for test
for i in range(710, len(test_files)):
    print(i, test_files[i])
    tr_id = test_files[i].split('/')[-1].split('_')[0]
    with open(test_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    X_seq_orig = np.asarray(data_dict['sequence'])

    # convert to liver codon format
    X_seq_standard = [liver_codon_to_number[ds06_number_to_codon[el]] for el in X_seq_orig]

    data_dict['sequence_standard'] = X_seq_standard

    # save the new file
    with open(test_files[i], 'wb') as f:
        pkl.dump(data_dict, f)
        f.close()