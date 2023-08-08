import pandas as pd 
import numpy as np 
import pickle as pkl 
from os import listdir
from os.path import isfile, join


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

codon_table_inv = dict(zip(codon_table.values(),codon_table.keys()))
# print(codon_table['NCT'], codon_table_inv[83])

codon_to_aa = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
        'NNG':'R', 'NNC':'T', 'NGT':'S', 'NGA':'R',
        'NNT':'Y', 'NGC':'S'
    }

aa_to_onehot = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    '_': 21
}

RNA_dict = {'A': [1,0,0,0,0], 'T': [0,1,0,0,0], 'G': [0,0,1,0,0], 'C': [0,0,0,1,0], 'N': [0,0,0,0,1]}

train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'

no_af2_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']

print("Starting")

conditions_list = []

# train data
mypath = train_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts:
        onlyfiles.append(x)

train_files = [mypath + '/' + f for f in onlyfiles]

print("Loaded Train")

# val data
mypath = val_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts:
        onlyfiles.append(x)

val_files = [mypath + '/' + f for f in onlyfiles]

print("Loaded Val")

# test data
mypath = test_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts:
        onlyfiles.append(x)

test_files = [mypath + '/' + f for f in onlyfiles]

print("Loaded Test")

print(f'Train Set: {len(train_files):5d} || Validation Set: {len(val_files):5d} || Test Set: {len(test_files): 5d}')

counter = 0
for f in train_files:
    counter += 1
    print(counter)

    # nts already done
    # codons first three and then we move to the next codon
    codon_encodings = [] # with the epa site
    # open file
    with open(f, 'rb') as file:
        data = pkl.load(file)
        file.close()

    # get the sequence
    seq = data['sequence']

    for i in range(2, len(seq)):
        codon_sample_encode = []
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i]-1] = 1
        codon_sample_encode.append(codon_one_hot)
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i-1]-1] = 1
        codon_sample_encode.append(codon_one_hot)  
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i-2]-1] = 1
        codon_sample_encode.append(codon_one_hot)

        # append the lists
        codon_sample_encode = np.array(codon_sample_encode)
        # concatenate the lists
        codon_sample_encode = np.concatenate(codon_sample_encode, axis=0)
        # print(codon_sample_encode.shape)

        codon_encodings.append(codon_sample_encode)
    
    codon_encodings = np.array(codon_encodings)

    data['codon_epa_encodings'] = codon_encodings

    # save the file
    with open(f, 'wb') as file:
        pkl.dump(data, file)
        file.close()

print("Done Train")

counter = 0
for f in val_files:
    counter += 1
    print(counter)

    # nts already done
    # codons first three and then we move to the next codon
    codon_encodings = [] # with the epa site
    # open file
    with open(f, 'rb') as file:
        data = pkl.load(file)
        file.close()

    # get the sequence
    seq = data['sequence']

    for i in range(2, len(seq)):
        codon_sample_encode = []
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i]-1] = 1
        codon_sample_encode.append(codon_one_hot)
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i-1]-1] = 1
        codon_sample_encode.append(codon_one_hot)  
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i-2]-1] = 1
        codon_sample_encode.append(codon_one_hot)

        # append the lists
        codon_sample_encode = np.array(codon_sample_encode)
        # concatenate the lists
        codon_sample_encode = np.concatenate(codon_sample_encode, axis=0)
        # print(codon_sample_encode.shape)

        codon_encodings.append(codon_sample_encode)
    
    codon_encodings = np.array(codon_encodings)

    data['codon_epa_encodings'] = codon_encodings

    # save the file
    with open(f, 'wb') as file:
        pkl.dump(data, file)
        file.close()

print("Done Val")

counter = 0
for f in test_files:
    counter += 1
    print(counter)

    # nts already done
    # codons first three and then we move to the next codon
    codon_encodings = [] # with the epa site
    # open file
    with open(f, 'rb') as file:
        data = pkl.load(file)
        file.close()

    # get the sequence
    seq = data['sequence']

    for i in range(2, len(seq)):
        codon_sample_encode = []
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i]-1] = 1
        codon_sample_encode.append(codon_one_hot)
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i-1]-1] = 1
        codon_sample_encode.append(codon_one_hot)  
        codon_one_hot = np.zeros(85)
        codon_one_hot[seq[i-2]-1] = 1
        codon_sample_encode.append(codon_one_hot)

        # append the lists
        codon_sample_encode = np.array(codon_sample_encode)
        # concatenate the lists
        codon_sample_encode = np.concatenate(codon_sample_encode, axis=0)
        # print(codon_sample_encode.shape)

        codon_encodings.append(codon_sample_encode)
    
    codon_encodings = np.array(codon_encodings)

    data['codon_epa_encodings'] = codon_encodings

    # save the file
    with open(f, 'wb') as file:
        pkl.dump(data, file)
        file.close()

print("Done Test")
