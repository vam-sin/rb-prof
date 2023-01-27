# libraries
import pickle as pkl 
import numpy as np
import random

# import data 
with open('../../data/rb_prof_Naef/processed_proper/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_finalNonEmpty.pkl', 'rb') as f:
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
        'TAC':'Y', 'TAT':'Y', 'TAA':'-', 'TAG':'-',
        'TGC':'C', 'TGT':'C', 'TGA':'-', 'TGG':'W',
        'TAA':'X', 'TGA':'X','TAG':'X'
    }

nts = ['A', 'T', 'G', 'C']

def sequence_gen(seq):
    protein = ""
    for j in range(0, len(seq), 3):
        codon = seq[j:j + 3]
        if len(codon) == 3:
            if codon in ['NNG','NNA','NNT','NNC','NAT','NAG','NAA','NAC','NTT','NTG','NTA','NTC','NGT','NGG','NGA','NGC','NCT','NCG','NCA','NCC']:
                if codon in ['NNG','NNA','NNT','NNC']:
                    codon = random.choice(nts) + codon[1:]
                    codon = codon[0] + random.choice(nts) + codon[2]
                else:
                    codon = random.choice(nts) + codon[1:]
                print(codon)
            
            protein+= table[codon]

    return protein
print(len(table.keys()))
protein_list = []

prot_file = open('../../data/rb_prof_Naef/processed_proper/seq_annot_final/translated_proteins.fasta', 'w')

for i in range(len(keys_list)):
    print('-'*89)
    print(i, len(keys_list), keys_list[i])
    input_seq = dict_seqCounts[keys_list[i]][0]
    # print(input_seq)
    protein_seq = sequence_gen(input_seq)
    protein_list.append(protein_seq)
    dict_seqCounts[keys_list[i]].append(protein_seq)
    prot_file.write('>' + str(keys_list[i]) + '\n')
    prot_file.write(str(protein_seq) + '\n')

prot_file.close()

print(len(protein_list))
with open('../../data/rb_prof_Naef/processed_proper/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_wProt.pkl', 'wb') as f:
    pkl.dump(dict_seqCounts, f)

'''
the whole transcript is converted into a protein sequence: have to figure out how to do the EPA sites thing
stop codons translated to X
'''