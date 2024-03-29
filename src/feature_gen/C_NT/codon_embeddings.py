import numpy as np
import itertools

number_to_codon = {idx+1:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G', 'N'], repeat=3))}
codon_to_number = {v: k for k, v in number_to_codon.items()}

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

codon_embeds = np.load('DNABERT_Codon_Embeds.npz', allow_pickle=True)['arr_0']
basic_64_embeds = codon_embeds[:64]
basic_64_codons = list(codon_table.keys())[:64]
basic_64_dict = {basic_64_codons[i]: basic_64_embeds[i] for i in range(len(basic_64_codons))}

total_codons_list = list(codon_to_number.keys())

dict_codon_embeds = {}

nts = ['A', 'T', 'G', 'C']

for i in range(len(total_codons_list)):
    print(i)
    if 'N' not in total_codons_list[i]:
        dict_codon_embeds[total_codons_list[i]] = basic_64_dict[total_codons_list[i]]
    else:
        codon_key = total_codons_list[i]
        permutations_list = []
        embeddings_list = []
        if codon_key[0] == 'N':
            if codon_key[1] == 'N':
                for x in range(len(nts)):
                    for y in range(len(nts)):
                        key = nts[x] + nts[y] + codon_key[2]
                        permutations_list.append(key)
            else:
                for x in range(len(nts)):
                    key = nts[x] + codon_key[1] + codon_key[2]
                    permutations_list.append(key)
        
        elif codon_key[1] == 'N':
            if codon_key[2] == 'N':
                for x in range(len(nts)):
                    for y in range(len(nts)):
                        key = codon_key[0] + nts[x] + nts[y]
                        permutations_list.append(key)
            else:
                for x in range(len(nts)):
                    key = codon_key[0] + nts[x] + codon_key[2]
                    permutations_list.append(key)

        else:
            for x in range(len(nts)):
                key = codon_key[0] + codon_key[1] + nts[x]
                permutations_list.append(key)
        
        for k in range(len(permutations_list)):
            embeddings_list.append(dict_codon_embeds[permutations_list[k]])

        
        embeddings_list = np.asarray(embeddings_list)
        mean_embed = np.mean(embeddings_list, axis=0)
        dict_codon_embeds[total_codons_list[i]] = mean_embed

print(len(list(dict_codon_embeds.keys())))

np.save('DNABERT_Codon_Embeds_NAv_Full.npy', dict_codon_embeds)