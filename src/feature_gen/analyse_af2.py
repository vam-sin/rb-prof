import pickle as pkl 
from Bio import SeqIO
from os import listdir
from os.path import isfile, join

with open('../models/regression/processed_keys/keys_proc_20c_60p.pkl', 'rb') as f:
    onlyfiles = pkl.load(f)

filepath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/'
af2_downloaded = [f for f in listdir(filepath) if isfile(join(filepath, f))]

onlyfiles_ = [x.replace('.npz','') for x in onlyfiles]
af2_downloaded_ = [x.replace('.pdb','') for x in af2_downloaded]

print(af2_downloaded_[0], onlyfiles_[0])
af2_downloaded_ = set(af2_downloaded_)
not_pred = []

for i in onlyfiles_:
    if i not in af2_downloaded_:
        not_pred.append(i)

print(len(not_pred))

cmd = ''

for i in not_pred:
    cmd += 'fasta_files/' + i + '.fasta,'

print(cmd)

'''
just need 172 more proteins
'''