# libraries
import numpy as np
import random
import copy
import time
from os import listdir
from os.path import isfile, join
import sys
from scipy.stats import pearsonr, spearmanr 
import pickle as pkl

train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'

def pcc_samples(tr1, tr2):
    file_1 = train_path + '/' + tr1
    file_2 = train_path + '/' + tr2
    # open pickle files and get count 
    with open(file_1, 'rb') as f:
        data_dict_1 = pkl.load(f)

    with open(file_2, 'rb') as f:
        data_dict_2 = pkl.load(f)
    
    y1 = np.log(1+data_dict_1['y'])
    y2 = np.log(1+data_dict_2['y'])

    pcc, _ = pearsonr(y1, y2)

    return pcc

mypath = train_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

transcripts_names = []

for i in range(len(onlyfiles_full)):
    transcripts_names.append(onlyfiles_full[i].split('_')[0])

transcripts_names = list(set(transcripts_names))

leu_ile_w_ile = []
leu_ile_val_w_val = []

for i in range(len(transcripts_names)):
    print(i, len(transcripts_names))
    transripts_w_conds_sample = []
    for j in range(len(onlyfiles_full)):
        if transcripts_names[i] in onlyfiles_full[j]:
            transripts_w_conds_sample.append(onlyfiles_full[j])

    leu_ile_tr = transcripts_names[i] + '_LEU-ILE_.pkl'
    leu_ile_val_tr = transcripts_names[i] + '_LEU-ILE-VAL_.pkl'
    ile_tr = transcripts_names[i] + '_ILE_.pkl'
    val_tr = transcripts_names[i] + '_VAL_.pkl'

    if leu_ile_tr in transripts_w_conds_sample and ile_tr in transripts_w_conds_sample:
        leu_ile_w_ile.append(pcc_samples(leu_ile_tr, ile_tr))
    
    if leu_ile_val_tr in transripts_w_conds_sample and val_tr in transripts_w_conds_sample:
        leu_ile_val_w_val.append(pcc_samples(leu_ile_val_tr, val_tr))

print('LEU-ILE vs ILE: ', np.mean(leu_ile_w_ile))
print('LEU-ILE-VAL vs VAL: ', np.mean(leu_ile_val_w_val))

'''
LEU-ILE vs ILE:  0.6237511826132706
LEU-ILE_VAL vs VAL:  0.8649506476176604
'''