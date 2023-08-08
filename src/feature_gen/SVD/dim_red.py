import pandas as pd 
import numpy as np 
import pickle as pkl 
from os import listdir
from os.path import isfile, join
from torch.linalg import svd
import torch

remove_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']

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

def conduct_svd_dimred(X, k):
    X = torch.from_numpy(X)
    U, S, _ = svd(X)
    dimred_x = torch.mm(U[:, :k], torch.diag(S[:k]))

    return dimred_x

def get_full_ft_vec(data_dict):
    nt = np.asarray(data_dict['nt'])
    cbert = np.asarray(data_dict['cbert'])
    t5 = np.asarray(data_dict['t5'])
    lem = np.asarray(data_dict['lem'])
    af2 = np.asarray(data_dict['AF2-SS'])
    geom = np.asarray(data_dict['geom'])

    len_ = geom.shape[0]
    nt = nt[:len_, :]
    cbert = cbert[:len_, :]
    lem = lem[:len_, :]

    print(nt.shape, cbert.shape, t5.shape, lem.shape, af2.shape, geom.shape)

    X = np.concatenate((nt, cbert, t5, lem, af2, geom), axis=1)

    return X

low_dim_ft = 64

# add af2 ss embeddings to train
for i in range(len(train_files)):
    print(i, train_files[i])
    tr_id = train_files[i].split('/')[-1].split('_')[0]
    if tr_id not in remove_transcripts:
        with open(train_files[i], 'rb') as f:
            data_dict = pkl.load(f)
            f.close()

        X = get_full_ft_vec(data_dict)
        dimred_X = conduct_svd_dimred(X, low_dim_ft)

        data_dict['svdimred'] = dimred_X
    
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
    
    X = get_full_ft_vec(data_dict)
    dimred_X = conduct_svd_dimred(X, low_dim_ft)

    data_dict['svdimred'] = dimred_X

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
    
    X = get_full_ft_vec(data_dict)
    dimred_X = conduct_svd_dimred(X, low_dim_ft)

    data_dict['svdimred'] = dimred_X

    # save the new file
    with open(val_files[i], 'wb') as f:
        pkl.dump(data_dict, f)
        f.close()
