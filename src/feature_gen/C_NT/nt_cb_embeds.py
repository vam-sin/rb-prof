'''
generates nt embeds + the c-bert embeds for the codons
'''

# libraries
import pickle as pkl 
import numpy as np 

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

RNA_dict = {'A': [1,0,0,0,0], 'T': [0,1,0,0,0], 'G': [0,0,1,0,0], 'C': [0,0,0,1,0], 'N': [0,0,0,0,1]}

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/seq_annot_final/fin_ILE.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

cbert_embeds = np.load('DNABERT_Codon_Embeds_NAv.npy', allow_pickle=True).item()

trascripts = list(dict_Tr_Seq.keys())
# print(dict_Tr_Seq[trascripts[0]])
for i in range(len(trascripts)):
    print(i, len(trascripts))
    codons_list = dict_Tr_Seq[trascripts[i]][0]
    codon_embeds = []
    nts_list = []
    for j in range(len(codons_list)):
        codon_3 = codon_table_inv[codons_list[j]]
        nts_list_sample = []
        for x in range(3):
            nts_list_sample += RNA_dict[codon_3[x]]
        nts_list.append(nts_list_sample)
        codon_embeds.append(cbert_embeds[codon_table_inv[codons_list[j]]])
    nts_list = np.asarray(nts_list)
    codon_embeds = np.asarray(codon_embeds)
    dict_Tr_Seq[trascripts[i]].append(nts_list)
    dict_Tr_Seq[trascripts[i]].append(codon_embeds)
    # print(len(codons_list), nts_list.shape, codon_embeds.shape)
    # break

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final_presplit/ILE_feats.pkl', 'wb') as f:
    pkl.dump(dict_Tr_Seq, f)