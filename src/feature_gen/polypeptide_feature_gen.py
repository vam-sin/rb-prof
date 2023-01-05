# libraries
import pickle as pkl 
import numpy as np

# import data 
with open('../../data/rb_prof_Naef/processed_data/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_final.pkl', 'rb') as f:
    dict_seqCounts = pkl.load(f)

keys_list = list(dict_seqCounts.keys())

table = {
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
    }

def sequence_gen(seq):
    protein = ""
    for j in range(0, len(seq), 3):
        codon = seq[j:j + 3]
        if len(codon) == 3:
            if codon in ['TAA', 'TGA', 'TAG']:
                print("BREAKING")
                break 

            protein+= table[codon]
    return protein

protein_list = []

for i in range(len(keys_list)):
    print('-'*89)
    print(i, len(keys_list))
    input_seq = dict_seqCounts[keys_list[i]][0]
    protein_seq = sequence_gen(input_seq)
    print(len(protein_seq), len(input_seq)/3)
    protein_list.append(protein_seq)

print(len(protein_list))

'''
the whole transcript is converted into a protein sequence: have to figure out how to do the EPA sites thing
'''