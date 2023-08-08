# libraries
import RNA
import numpy as np 
import pickle as pkl 
from os import listdir
from os.path import isfile, join
from sknetwork.embedding import Spectral

codon_table = {
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

# inverse codon table
inv_codon_table = {v: k for k, v in codon_table.items()}

# spectral decomposition
spectral_decomp = Spectral(5)

def getRNASeq(codon_seq):
    rna_seq = ''
    for i in range(len(codon_seq)):
        rna_seq += inv_codon_table[codon_seq[i]]

    return rna_seq

def get_LEM(ss):
    len_ss = len(ss)
    # make adj matrix
    adj = np.zeros((len_ss, len_ss))

    # every nt connected to the one right after it
    for j in range(len_ss-1):
        adj[j][j+1] = 1.0
        adj[j+1][j] = 1.0

    # the loops
    stack = []
    for j in range(len_ss):
        if ss[j] == '(':
            stack.append(j)
        elif ss[j] == ')':
            conn_1 = j 
            conn_2 = stack.pop()
            adj[conn_1][conn_2] = 1.0
            adj[conn_2][conn_1] = 1.0
        else:
            pass 

    adj = np.asarray(adj)

    embeds = spectral_decomp.fit_transform(adj)
    return embeds

def get_features(codon_seq):
    # get rna seq from codon seq
    rna_seq = getRNASeq(codon_seq)
    # get rna structure from rna seq
    struc, mf = RNA.fold(rna_seq)
    # get rna structure features
    struc_features = get_LEM(struc)
    # convert to numpy array and make sets of 3
    struc_features = np.asarray(struc_features)
    codon_struc_features = []
    for i in range(len(struc_features)):
        if i%3 == 0:
            # concatenate 3 features
            conc_lem = np.concatenate((struc_features[i], struc_features[i+1], struc_features[i+2]))
            codon_struc_features.append(conc_lem)
    
    codon_struc_features = np.asarray(codon_struc_features)

    return codon_struc_features

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
train_files = [mypath + '/' + f for f in onlyfiles]

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_files = [mypath + '/' + f for f in onlyfiles]

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
val_files = [mypath + '/' + f for f in onlyfiles]

# add LEM embeddings to train
for i in range(len(train_files)):
    print(i)
    with open(train_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    X_seq = np.asarray(data_dict['sequence'])

    # get LEM embeddings
    lem_embeds = get_features(X_seq)

    # check if the number of embeddings is the same as the sequence length
    assert lem_embeds.shape[0] == X_seq.shape[0]
    print(lem_embeds.shape, X_seq.shape)

    # update the data dict
    data_dict['LEM'] = lem_embeds

    # save the new file
    with open(train_files[i], 'wb') as f:
        pkl.dump(data_dict, f)
        f.close()

# add LEM embeddings to val
for i in range(len(val_files)):
    print(i)
    with open(val_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    X_seq = np.asarray(data_dict['sequence'])

    # get LEM embeddings
    lem_embeds = get_features(X_seq)

    # check if the number of embeddings is the same as the sequence length
    assert lem_embeds.shape[0] == X_seq.shape[0]
    print(lem_embeds.shape, X_seq.shape)

    # update the data dict
    data_dict['LEM'] = lem_embeds

    # save the new file
    with open(val_files[i], 'wb') as f:
        pkl.dump(data_dict, f)
        f.close()

# add LEM embeddings to test
for i in range(len(test_files)):
    print(i)
    with open(test_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    X_seq = np.asarray(data_dict['sequence'])

    # get LEM embeddings
    lem_embeds = get_features(X_seq)

    # check if the number of embeddings is the same as the sequence length
    assert lem_embeds.shape[0] == X_seq.shape[0]
    print(lem_embeds.shape, X_seq.shape)

    # update the data dict
    data_dict['LEM'] = lem_embeds

    # save the new file
    with open(test_files[i], 'wb') as f:
        pkl.dump(data_dict, f)
        f.close()

