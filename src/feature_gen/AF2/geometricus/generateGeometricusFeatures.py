'''
takes in the PDB files to generate the geometricus features
'''

import pandas as pd 
import numpy as np 
import prody
from geometricus import MomentInvariants, SplitType
import pickle as pkl 
from os import listdir
from os.path import isfile, join

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

remove_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']

# train files
for i in range(len(train_files)):
    print(i, train_files[i])
    tr_id = train_files[i].split('/')[-1].split('_')[0]

    if tr_id not in remove_transcripts:
        with open(train_files[i], 'rb') as f:
            data_dict = pkl.load(f)
            f.close()
        
        X_seq = np.asarray(data_dict['sequence'])
        X_T5 = np.asarray(data_dict['t5'])

        pdb_file_name = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/' + tr_id + '.pdb'

        pdb = prody.parsePDB(pdb_file_name)

        kmer_inv = MomentInvariants.from_prody_atomgroup(tr_id, pdb, split_type=SplitType.KMER, split_size=16)
        kmer_inv = np.asarray(kmer_inv.moments)

        rad_inv = MomentInvariants.from_prody_atomgroup(tr_id, pdb, split_type=SplitType.RADIUS, split_size=10)
        rad_inv = np.asarray(rad_inv.moments)

        # normalize kmer_inv and rad_inv
        kmer_inv = (kmer_inv - np.mean(kmer_inv, axis=0)) / np.std(kmer_inv, axis=0)
        rad_inv = (rad_inv - np.mean(rad_inv, axis=0)) / np.std(rad_inv, axis=0)

        # append the features to each other
        geom_ft = np.concatenate((kmer_inv, rad_inv), axis=1)

        # add the features to the data_dict
        data_dict['geom'] = geom_ft

        # save the data_dict
        with open(train_files[i], 'wb') as f:
            pkl.dump(data_dict, f)
            f.close()

# val files
for i in range(len(val_files)):
    print(i, val_files[i])
    tr_id = val_files[i].split('/')[-1].split('_')[0]

    if tr_id not in remove_transcripts:
        with open(val_files[i], 'rb') as f:
            data_dict = pkl.load(f)
            f.close()
        
        X_seq = np.asarray(data_dict['sequence'])
        X_T5 = np.asarray(data_dict['t5'])

        pdb_file_name = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/' + tr_id + '.pdb'

        pdb = prody.parsePDB(pdb_file_name)

        kmer_inv = MomentInvariants.from_prody_atomgroup(tr_id, pdb, split_type=SplitType.KMER, split_size=16)
        kmer_inv = np.asarray(kmer_inv.moments)

        rad_inv = MomentInvariants.from_prody_atomgroup(tr_id, pdb, split_type=SplitType.RADIUS, split_size=10)
        rad_inv = np.asarray(rad_inv.moments)

        # normalize kmer_inv and rad_inv
        kmer_inv = (kmer_inv - np.mean(kmer_inv, axis=0)) / np.std(kmer_inv, axis=0)
        rad_inv = (rad_inv - np.mean(rad_inv, axis=0)) / np.std(rad_inv, axis=0)

        # append the features to each other
        geom_ft = np.concatenate((kmer_inv, rad_inv), axis=1)

        # add the features to the data_dict
        data_dict['geom'] = geom_ft

        # save the data_dict
        with open(val_files[i], 'wb') as f:
            pkl.dump(data_dict, f)
            f.close()

# test files
for i in range(len(test_files)):
    print(i, test_files[i])
    tr_id = test_files[i].split('/')[-1].split('_')[0]

    if tr_id not in remove_transcripts:
        with open(test_files[i], 'rb') as f:
            data_dict = pkl.load(f)
            f.close()
        
        X_seq = np.asarray(data_dict['sequence'])
        X_T5 = np.asarray(data_dict['t5'])

        pdb_file_name = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/' + tr_id + '.pdb'

        pdb = prody.parsePDB(pdb_file_name)

        kmer_inv = MomentInvariants.from_prody_atomgroup(tr_id, pdb, split_type=SplitType.KMER, split_size=16)
        kmer_inv = np.asarray(kmer_inv.moments)

        rad_inv = MomentInvariants.from_prody_atomgroup(tr_id, pdb, split_type=SplitType.RADIUS, split_size=10)
        rad_inv = np.asarray(rad_inv.moments)

        # normalize kmer_inv and rad_inv
        kmer_inv = (kmer_inv - np.mean(kmer_inv, axis=0)) / np.std(kmer_inv, axis=0)
        rad_inv = (rad_inv - np.mean(rad_inv, axis=0)) / np.std(rad_inv, axis=0)

        # append the features to each other
        geom_ft = np.concatenate((kmer_inv, rad_inv), axis=1)

        # add the features to the data_dict
        data_dict['geom'] = geom_ft

        # save the data_dict
        with open(test_files[i], 'wb') as f:
            pkl.dump(data_dict, f)
            f.close()