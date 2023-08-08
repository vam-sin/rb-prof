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

out_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/mimo_data/'

cbert_embeds = np.load('DNABERT_Codon_Embeds_NAv.npy', allow_pickle=True).item()

# load the txt file from ribomimo
with open('../../../repos/RiboMIMO/data/Subtelny14.txt') as f:
    lines = f.readlines()
    lines = [x.strip() for x in lines]

gene_names = []

for i in range(len(lines)):
    print(i, len(lines))
    # make a pickle file for each gene
    if i%3==0:
        # mohammed16
        # gene_names.append(lines[i].replace('>', '').split('|')[1].split(' ')[0])
        # mohammed19-1
        # gene_names.append(lines[i].replace('>', ''))
        # subtelny14
        gene_names.append(lines[i].replace('>', '').replace(' ', ''))
        data_dict = {}
        data_dict['gene'] = gene_names[-1]
        with open(out_folder + gene_names[-1] + '.pkl', 'wb') as f:
            pkl.dump(data_dict, f)
    elif i%3==1:
        # open the pickle file and add the sequence
        with open(out_folder + gene_names[-1] + '.pkl', 'rb') as f:
            data_dict = pkl.load(f)
        
        # codon_split_seq = lines[i].split('\t')
        # codon_split_seq = lines[i].split('

        try:
            codon_split_seq = lines[i].split('\t')
            codon_split_seq = [x.replace(' ', '') for x in codon_split_seq]
        except:
            print("except")
            codon_split_seq = lines[i].split(' ')

        for x in range(len(codon_split_seq)):
            codon_split_seq[x] = codon_table[codon_split_seq[x]]

        data_dict['sequence'] = codon_split_seq

        # make nt and codon embeds
        nts_list = []
        codon_embeds = []
        
        for j in range(len(codon_split_seq)):
            codon_3 = codon_table_inv[codon_split_seq[j]]
            nts_list_sample = []
            for x in range(3):
                nts_list_sample += RNA_dict[codon_3[x]]
            nts_list.append(nts_list_sample)
            codon_embeds.append(cbert_embeds[codon_table_inv[codon_split_seq[j]]])
        
        data_dict['nt'] = np.asarray(nts_list)
        data_dict['cbert'] = np.asarray(codon_embeds)

        # save the pickle file
        with open(out_folder + gene_names[-1] + '.pkl', 'wb') as f:
            pkl.dump(data_dict, f)

    else:
        counts = lines[i].split('\t')
        counts = [float(x) for x in counts]
        
        # open the pickle file and add the counts
        with open(out_folder + gene_names[-1] + '.pkl', 'rb') as f:
            data_dict = pkl.load(f)
        
        data_dict['y'] = counts

        # save the pickle file
        with open(out_folder + gene_names[-1] + '.pkl', 'wb') as f:
            pkl.dump(data_dict, f)
        