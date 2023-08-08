'''
checks if all the AF2 structures have been predicted well
'''

import pickle as pkl 
from Bio import SeqIO
from os import listdir
from os.path import isfile, join
import pandas as pd

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
train_files = [mypath + '/' + f for f in onlyfiles]

transcripts_unique_train = []
for i in range(len(train_files)):
    transcripts_unique_train.append(train_files[i].split('/')[-1].split('_')[0])
transcripts_unique_train = list(set(transcripts_unique_train))

# val data
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
val_files = [mypath + '/' + f for f in onlyfiles]

transcripts_unique_val = []
for i in range(len(val_files)):
    transcripts_unique_val.append(val_files[i].split('/')[-1].split('_')[0])
transcripts_unique_val = list(set(transcripts_unique_val))

# test data
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_files = [mypath + '/' + f for f in onlyfiles]

transcripts_unique_test = []
for i in range(len(test_files)):
    transcripts_unique_test.append(test_files[i].split('/')[-1].split('_')[0])
transcripts_unique_test = list(set(transcripts_unique_test))

# all data
transcripts_unique = transcripts_unique_train + transcripts_unique_val + transcripts_unique_test

print(len(transcripts_unique))

filepath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/'
af2_downloaded = [f for f in listdir(filepath) if isfile(join(filepath, f))]

print(len(af2_downloaded))

# get codon sequence lengths
ds = pd.read_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/protein_sequences.csv')
ds_tr = list(ds['transcript_label'])
prot_seq_ds = list(ds['protein_seq_without_stop'])
lenghts_ds = [len(x) for x in prot_seq_ds]

# make dict from ds
ds_dict = {}
for i in range(len(ds_tr)):
    ds_dict[ds_tr[i]] = lenghts_ds[i]


redo_preds_file = open('redo.txt', 'w')
# go over the pdbs and check if the number of residues is same as the length of the protein sequence
for i in range(len(transcripts_unique)):
    try:
        print(i)
        # open the pdb file
        transcript_sample_id = transcripts_unique[i].split('/')[-1].split('_')[0]
        print(transcript_sample_id)
        # get number of residues from the pdb file
        pdb_file = filepath + transcript_sample_id + '.pdb'
        for record in SeqIO.parse(pdb_file, "pdb-atom"):
            pdb_seq_len = len(str(record.seq))

        print(pdb_seq_len, ds_dict[transcript_sample_id])

        if pdb_seq_len != ds_dict[transcript_sample_id]:
            redo_preds_file.write(transcript_sample_id + '\n')
    except:
        redo_preds_file.write(transcript_sample_id + '\n')

    # break
