# library imports
import pickle as pkl
import pandas as pd
from sklearn import preprocessing
import itertools

# codon to number
id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
print(id_to_codon)
codon_to_id = {v:k for k,v in id_to_codon.items()}

prev_codon_table = {
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

prev_id_to_codon = {v:k for k,v in prev_codon_table.items()}

folder_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/seq_annot_raw/'

files = ['codon_annot_CTRL.pkl', 'codon_annot_ILE.pkl', 'codon_annot_LEU.pkl', 'codon_annot_LEU_ILE.pkl', 'codon_annot_VAL.pkl', 'codon_annot_LEU_ILE_VAL.pkl']

filenames = [folder_path + file for file in files]

print(filenames)

file = filenames[5]

with open(file, 'rb') as f:
    data = pkl.load(f)

data_keys = list(data.keys())

print(len(data_keys))

df_transcript = []
df_gene = []
df_sequence = []
df_annots = []
df_perc_annots = []

count = 0
for key in data_keys:
    print(count)
    count += 1
    seq = data[key][0]
    # change sequence from prev codon ids to new codon ids
    seq = [prev_id_to_codon[i] for i in seq]
    n_flag = False
    # check if any codon with N in it
    for x in seq:
        if 'N' in x:
            print('N in sequence', key, 'is removed')
            n_flag = True
            break

    if n_flag == False:
        seq = [codon_to_id[i] for i in seq]
        annots = data[key][1]
        gene = data[key][2]
        num_non_zero_in_annots = len([i for i in annots if i != 0.0])
        perc_non_zero_in_annots = num_non_zero_in_annots/len(annots)

        # add to dataframe
        df_transcript.append(key)
        df_gene.append(gene)
        df_sequence.append(seq)
        df_annots.append(annots)
        df_perc_annots.append(perc_non_zero_in_annots)  

# make dataframe
final_df = pd.DataFrame(list(zip(df_transcript, df_gene, df_sequence, df_annots, df_perc_annots)), columns = ['transcript', 'gene', 'sequence', 'annotations', 'perc_non_zero_annots'])

print(final_df)

# save dataframe
final_df.to_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/LEU_ILE_VAL.csv', index=False)

'''
CTRL: 17184
ILE: 16269
LEU: 16051
LEU_ILE: 15443
VAL: 12665
LEU_ILE_VAL: 12965
'''


