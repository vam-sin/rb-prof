'''
takes in the PDB files and generates the AF2 SS features
'''

import pandas as pd 
import numpy as np 
import pickle as pkl 
from os import listdir
from os.path import isfile, join
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

# Structures
# H,G,I: 1 
# T: 2 (T, S)
# S: 3
# B: 4
# E: 5
# - 6
# Exception: 7

remove_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']

p = PDBParser()

def get_ss_ft(pdb_file):
    ssf = []
    pdb_file_name = pdb_file.split('/')[-1]
    structure = p.get_structure(pdb_file_name, pdb_file)
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    sec_structure = ''
    for z in range(len(dssp)):
        a_key = list(dssp.keys())[z]
        sec_structure += dssp[a_key][2]
    # print(sec_structure)
    for z in range(len(sec_structure)):
        ssf_sample = np.zeros((5))
        if sec_structure[z] == 'H' or sec_structure[z] == 'G' or sec_structure[z] == 'I' or sec_structure[z] == 'P': # alpha helix
            ssf_sample[0] = 1
        if sec_structure[z] == 'T' or sec_structure[z] == 'S': # turn/bend
            ssf_sample[1] = 1
        if sec_structure[z] == 'B':
            ssf_sample[2] = 1
        if sec_structure[z] == 'E':
            ssf_sample[3] = 1
        if sec_structure[z] not in ['H', 'G', 'I', 'P', 'T', 'S', 'B', 'E']:
            ssf_sample[4] = 1
        ssf.append(ssf_sample)

    ssf = np.asarray(ssf)
    return ssf

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
train_files = [mypath + '/' + f for f in onlyfiles]

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_files = [mypath + '/' + f for f in onlyfiles]

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
val_files = [mypath + '/' + f for f in onlyfiles]

print("Load complete")

# add af2 ss embeddings to train
for i in range(len(train_files)):
    print(i, train_files[i])
    tr_id = train_files[i].split('/')[-1].split('_')[0]
    if tr_id not in remove_transcripts:
        with open(train_files[i], 'rb') as f:
            data_dict = pkl.load(f)
            f.close()
        
        X_seq = np.asarray(data_dict['sequence'])
        X_T5 = np.asarray(data_dict['t5'])

        pdb_file = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/' + tr_id + '.pdb'
        try:
            ss_ft = get_ss_ft(pdb_file)

            # concatenate
            try:    
                X_new = np.concatenate((X_T5, ss_ft), axis=1)
                # print(ss_ft)
                data_dict['AF2-SS'] = ss_ft
            except:
                print('Error in dims: ', train_files[i], ss_ft.shape, X_T5.shape)
        except:
            print('Error in file: ', train_files[i])

        # save the new file
        with open(train_files[i], 'wb') as f:
            pkl.dump(data_dict, f)
            f.close()

# add af2 ss embeddings to test
for i in range(len(test_files)):
    print(i, test_files[i])
    with open(test_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    X_seq = np.asarray(data_dict['sequence'])
    X_T5 = np.asarray(data_dict['t5'])
    tr_id = test_files[i].split('/')[-1].split('_')[0]
    if tr_id not in remove_transcripts:
        pdb_file = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/' + tr_id + '.pdb'
        try:
            ss_ft = get_ss_ft(pdb_file)
            # print(ss_ft.shape, X_T5.shape)

            # concatenate
            try:    
                X_new = np.concatenate((X_T5, ss_ft), axis=1)
                data_dict['AF2-SS'] = ss_ft
            except:
                print('Error in dims: ', test_files[i], ss_ft.shape, X_T5.shape)
        except:
            print('Error in file: ', test_files[i])

        # save the new file
        with open(test_files[i], 'wb') as f:
            pkl.dump(data_dict, f)
            f.close()

# add af2 ss embeddings to val
for i in range(len(val_files)):
    print(i, val_files[i])
    with open(val_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    X_seq = np.asarray(data_dict['sequence'])
    X_T5 = np.asarray(data_dict['t5'])
    tr_id = val_files[i].split('/')[-1].split('_')[0]
    if tr_id not in remove_transcripts:
        pdb_file = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/' + tr_id + '.pdb'
        try:
            ss_ft = get_ss_ft(pdb_file)
            # print(ss_ft.shape, X_T5.shape)

            # concatenate
            try:    
                X_new = np.concatenate((X_T5, ss_ft), axis=1)
                data_dict['AF2-SS'] = ss_ft
            except:
                print('Error in dims: ', val_files[i], ss_ft.shape, X_T5.shape)
        except:
            print('Error in file: ', val_files[i])

        # save the new file
        with open(val_files[i], 'wb') as f:
            pkl.dump(data_dict, f)
            f.close()
