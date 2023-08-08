# libraries
import pickle as pkl 
import numpy as np
from Bio import SeqIO 
import pandas as pd
import random

# import data 
with open('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_raw/dataset_ProcFeats_fixed_prot.pkl', 'rb') as f:
    data = pkl.load(f)

# keys_list = list(data.keys())

# table = {
#         'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
#         'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
#         'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
#         'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                
#         'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
#         'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
#         'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
#         'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
#         'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
#         'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
#         'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
#         'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
#         'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
#         'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
#         'TAC':'Y', 'TAT':'Y', 'TAA':'-', 'TAG':'-',
#         'TGC':'C', 'TGT':'C', 'TGA':'-', 'TGG':'W',
#         'TAA':'X', 'TGA':'X','TAG':'X'
#     }

# nts = ['A', 'T', 'G', 'C']

# # mouse protein full set
# seq_list = []
# desc_uni = []

# with open("../../data/mouse_prots/uniprot_mus_musculus_S100.fasta") as handle:
#     for record in SeqIO.parse(handle, "fasta"):
#         seq_list.append(str(record.seq))
#         desc_uni.append(record.description.split('|')[1])

# mouse_proteins_ds = dict(zip(seq_list, desc_uni))

# def get_uniprot_match(prot):
#     try:
#         return mouse_proteins_ds[prot]
#     except:
#         return ''

# def sequence_gen(seq):
#     protein = ""
#     for j in range(0, len(seq), 3):
#         codon = seq[j:j + 3]
#         if len(codon) == 3:
#             if codon in ['NNG','NNA','NNT','NNC','NAT','NAG','NAA','NAC','NTT','NTG','NTA','NTC','NGT','NGG','NGA','NGC','NCT','NCG','NCA','NCC']:
                
#                 if codon in ['NNG','NNA','NNT','NNC']:
#                     codon = random.choice(nts) + codon[1:]
#                     codon = codon[0] + random.choice(nts) + codon[2]
#                 else:
#                     codon = random.choice(nts) + codon[1:]
#                 print(codon)
            
#             protein+= table[codon]

#     len_prot = len(protein)
#     edit_seq = protein
#     if protein[len_prot-1] == 'X':
#         edit_seq = protein[:len_prot-1]
#     uniprot_match_id = get_uniprot_match(edit_seq)
    
#     return protein, uniprot_match_id

# print(len(table.keys()))

# protein_list = []
# count = 0
# for i in range(len(keys_list)):
#     print('-'*89)
#     print(i, len(keys_list), keys_list[i])
#     key_ = keys_list[i]
#     input_seq = data[key_]['RNA_Seq']
#     ambi_flag = 0
#     protein_seq, uniprot_match_fin = sequence_gen(input_seq)
#     protein_list.append(protein_seq)
#     data[key_]['Protein_Seq'] = protein_seq
#     if uniprot_match_fin != '':
#         data[key_]['Uniprot'] = uniprot_match_fin
#         count += 1

# print(len(protein_list))
# print(count)

# with open('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_raw/dataset_ProcFeats_fixed_prot.pkl', 'wb') as f:
#     pkl.dump(data, f)

keys_list = list(data.keys())

prot_list = []

for i in range(len(keys_list)):
    prot_list.append(data[keys_list[i]]['Protein_Seq'])

df = pd.DataFrame(list(zip(keys_list, prot_list)),
               columns =['Record', 'Sequence'])

df.to_csv('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_raw/Prot_FINAL.csv')

'''
the whole transcript is converted into a protein sequence: have to figure out how to do the EPA sites thing
stop codons translated to X
'''