import numpy as np 
import pandas as pd
from os import listdir
import pickle as pkl

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
codon_table_inv = {v: k for k, v in codon_table.items()}

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

val_files = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val/'
test_files = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test/'
train_files = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train/'

val_samples = [f for f in listdir(val_files) if f.endswith('.pkl')]
test_samples = [f for f in listdir(test_files) if f.endswith('.pkl')]
train_samples = [f for f in listdir(train_files) if f.endswith('.pkl')]

print(len(val_samples), len(test_samples), len(train_samples))

protein_sequences = []
transcript_labels = []

print("Train")
count = 0
# convert codon to aa
for sample in train_samples:
    print(count)
    count += 1
    f_name = train_files + sample
    with open(f_name, 'rb') as f:
        loaded_pkl_file = pkl.load(f)
    codon_seq = loaded_pkl_file['sequence']
    # print(codon_seq)
    prot_sample = ''
    for i in range(len(codon_seq)):
        prot_sample = prot_sample + codon_to_aa[codon_table_inv[codon_seq[i]]]
    protein_sequences.append(prot_sample)
    transcript_labels.append(sample.split('_')[0])

    # print(sample.split('_')[0], prot_sample)
    # break

print("Test")
for sample in test_samples:
    print(count)
    count += 1
    f_name = test_files + sample
    with open(f_name, 'rb') as f:
        loaded_pkl_file = pkl.load(f)
    codon_seq = loaded_pkl_file['sequence']
    # print(codon_seq)
    prot_sample = ''
    for i in range(len(codon_seq)):
        prot_sample = prot_sample + codon_to_aa[codon_table_inv[codon_seq[i]]]
    protein_sequences.append(prot_sample)
    transcript_labels.append(sample.split('_')[0])

    # print(sample.split('_')[0], prot_sample)
    # break

print("Val")
for sample in val_samples:
    print(count)
    count += 1
    f_name = val_files + sample
    with open(f_name, 'rb') as f:
        loaded_pkl_file = pkl.load(f)
    codon_seq = loaded_pkl_file['sequence']
    # print(codon_seq)
    prot_sample = ''
    for i in range(len(codon_seq)):
        prot_sample = prot_sample + codon_to_aa[codon_table_inv[codon_seq[i]]]
    protein_sequences.append(prot_sample)
    transcript_labels.append(sample.split('_')[0])

    # print(sample.split('_')[0], prot_sample)
    # break

print(len(protein_sequences), len(transcript_labels))

# make dataframe from lists
protein_sequences_without_stop = []
for i in range(len(protein_sequences)):
    protein_sequences_without_stop.append(protein_sequences[i].replace('_', ''))

df = pd.DataFrame(list(zip(protein_sequences, protein_sequences_without_stop, transcript_labels)), columns =['protein_seq', 'protein_seq_without_stop', 'transcript_label'])
df.to_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/protein_sequences.csv', index=False)