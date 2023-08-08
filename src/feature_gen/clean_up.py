# libraries
import numpy as np
from sklearn.model_selection import train_test_split
import torch 
from torch import nn
from scipy.stats import pearsonr, spearmanr 
from os import listdir
from os.path import isfile, join
import math
import pickle as pkl 
from torch.autograd import Variable
from torch.utils.data import Dataset


# train data
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
train_files = [mypath + '/' + f for f in onlyfiles]

print("Loaded Train")

# val data
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
val_files = [mypath + '/' + f for f in onlyfiles]

print("Loaded Val")

# test data
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_files = [mypath + '/' + f for f in onlyfiles]

print("Loaded Test")

keys_to_keep = []

for x in train_files:
    # load data
    with open(x, 'rb') as f:
        data_dict = pkl.load(f)
    
    keys = list(data_dict.keys())
    X_nt_cbert = data_dict['X']
    nt = X_nt_cbert[:,:15]
    cbert = X_nt_cbert[:,15:15+768]
    conds = X_nt_cbert[:,15+768:15+768+20]
    depr_vec = X_nt_cbert[:,15+768+20:15+768+20+1]

    X_nt_cbert_t5 = data_dict['X_T5']
    t5 = X_nt_cbert_t5[:,15+768:15+768+1024]

    lem = data_dict['LEM']
    # print(lem.shape)

    mlm_cdna_nt = data_dict['X_MLM_cDNA_NT']
    # remove conds from this
    mlm_cdna_nt = mlm_cdna_nt[:,:1536]
    # print(mlm_cdna_nt.shape)

    mlm_cdna_nt_idai = data_dict['X_MLM_cDNA_NT_IDAI']
    # remove conds from this
    mlm_cdna_nt_idai = mlm_cdna_nt_idai[:,:7680]
    # print(mlm_cdna_nt_idai.shape)

    clean_dict = {}
    clean_dict['sequence'] = data_dict['sequence']
    clean_dict['gene'] = data_dict['gene']
    clean_dict['y'] = data_dict['y']
    clean_dict['nt'] = nt 
    clean_dict['cbert'] = cbert
    clean_dict['conds'] = conds
    clean_dict['depr_vec'] = depr_vec
    clean_dict['t5'] = t5
    clean_dict['lem'] = lem
    clean_dict['mlm_cdna_nt_pbert'] = mlm_cdna_nt
    clean_dict['mlm_cdna_nt_idai'] = mlm_cdna_nt_idai

    # save the new file
    with open(x, 'wb') as f:
        pkl.dump(clean_dict, f)

for x in val_files:
    # load data
    with open(x, 'rb') as f:
        data_dict = pkl.load(f)
    
    keys = list(data_dict.keys())
    X_nt_cbert = data_dict['X']
    nt = X_nt_cbert[:,:15]
    cbert = X_nt_cbert[:,15:15+768]
    conds = X_nt_cbert[:,15+768:15+768+20]
    depr_vec = X_nt_cbert[:,15+768+20:15+768+20+1]

    X_nt_cbert_t5 = data_dict['X_T5']
    t5 = X_nt_cbert_t5[:,15+768:15+768+1024]

    lem = data_dict['LEM']
    # print(lem.shape)

    mlm_cdna_nt = data_dict['X_MLM_cDNA_NT']
    # remove conds from this
    mlm_cdna_nt = mlm_cdna_nt[:,:1536]
    # print(mlm_cdna_nt.shape)

    mlm_cdna_nt_idai = data_dict['X_MLM_cDNA_NT_IDAI']
    # remove conds from this
    mlm_cdna_nt_idai = mlm_cdna_nt_idai[:,:7680]
    # print(mlm_cdna_nt_idai.shape)

    clean_dict = {}
    clean_dict['sequence'] = data_dict['sequence']
    clean_dict['gene'] = data_dict['gene']
    clean_dict['y'] = data_dict['y']
    clean_dict['nt'] = nt 
    clean_dict['cbert'] = cbert
    clean_dict['conds'] = conds
    clean_dict['depr_vec'] = depr_vec
    clean_dict['t5'] = t5
    clean_dict['lem'] = lem
    clean_dict['mlm_cdna_nt_pbert'] = mlm_cdna_nt
    clean_dict['mlm_cdna_nt_idai'] = mlm_cdna_nt_idai

    # save the new file
    with open(x, 'wb') as f:
        pkl.dump(clean_dict, f)

for x in test_files:
    # load data
    with open(x, 'rb') as f:
        data_dict = pkl.load(f)
    
    keys = list(data_dict.keys())
    X_nt_cbert = data_dict['X']
    nt = X_nt_cbert[:,:15]
    cbert = X_nt_cbert[:,15:15+768]
    conds = X_nt_cbert[:,15+768:15+768+20]
    depr_vec = X_nt_cbert[:,15+768+20:15+768+20+1]

    X_nt_cbert_t5 = data_dict['X_T5']
    t5 = X_nt_cbert_t5[:,15+768:15+768+1024]

    lem = data_dict['LEM']
    # print(lem.shape)

    mlm_cdna_nt = data_dict['X_MLM_cDNA_NT']
    # remove conds from this
    mlm_cdna_nt = mlm_cdna_nt[:,:1536]
    # print(mlm_cdna_nt.shape)

    mlm_cdna_nt_idai = data_dict['X_MLM_cDNA_NT_IDAI']
    # remove conds from this
    mlm_cdna_nt_idai = mlm_cdna_nt_idai[:,:7680]
    # print(mlm_cdna_nt_idai.shape)

    clean_dict = {}
    clean_dict['sequence'] = data_dict['sequence']
    clean_dict['gene'] = data_dict['gene']
    clean_dict['y'] = data_dict['y']
    clean_dict['nt'] = nt 
    clean_dict['cbert'] = cbert
    clean_dict['conds'] = conds
    clean_dict['depr_vec'] = depr_vec
    clean_dict['t5'] = t5
    clean_dict['lem'] = lem
    clean_dict['mlm_cdna_nt_pbert'] = mlm_cdna_nt
    clean_dict['mlm_cdna_nt_idai'] = mlm_cdna_nt_idai

    # save the new file
    with open(x, 'wb') as f:
        pkl.dump(clean_dict, f)