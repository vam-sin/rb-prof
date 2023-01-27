from gensim.models import KeyedVectors
import numpy as np

filepath = '../../repos/dna2vec/results/dna2vec-k3to3-withN.w2v'
mk_model = KeyedVectors.load_word2vec_format(filepath, binary=False)

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
        'TGC':61, 'TGT':62, 'TGA':63, 'TGG':64,
        '-': 65, 'NNG': 66, 'NGG': 67, 'NNT': 68,
        'NTG': 69, 'NAC': 70, 'NNC': 71, 'NCC': 72,
        'NGC': 73, 'NCA': 74, 'NGA': 75, 'NNA': 76,
        'NAG': 77, 'NTC': 78, 'NAT': 79, 'NGT': 80,
        'NCG': 81, 'NTT': 82, 'NCT': 83, 'NAA': 84,
        'NTA': 85
    }

codons = list(codon_table.keys())
print(codons)


key2index = mk_model.key_to_index
embeds = {}

for i in range(len(codons)):
    if codons[i] not in ['-']:
        index_ = key2index[codons[i]]
        print(i, codons[i], index_)
        embeds[codons[i]] = mk_model[index_]

# print(len(mk_model[0]))

embeds['-'] = np.zeros((100))
# embeds['NNG'] = (embeds['ACG'] + embeds['CTG'] + embeds['CCG'] + embeds['CGG'] + embeds['GTG'] + embeds['GCG'] + embeds['GGG'] + embeds['TCG'] + embeds['ATG'] + embeds['AAG'] + embeds['AGG'] + embeds['CAG'] + embeds['GAG'] + embeds['TTG'] + embeds['TAG'] + embeds['TGG']) / 16
# embeds['NGG'] = (embeds['AGG'] + embeds['CGG'] + embeds['TGG'] + embeds['GGG']) / 4
# embeds['NNT'] = (embeds['ACT'] + embeds['CTT'] + embeds['CCT'] + embeds['CGT'] + embeds['GTT'] + embeds['GCT'] + embeds['GGT'] + embeds['TCT'] + embeds['ATT'] + embeds['AAT'] + embeds['AGT'] + embeds['CAT'] + embeds['GAT'] + embeds['TTT'] + embeds['TAT'] + embeds['TGT']) / 16
# embeds['NTG'] = (embeds['ATG'] + embeds['CTG'] + embeds['TTG'] + embeds['GTG']) / 4
# embeds['NAC'] = (embeds['AAC'] + embeds['CAC'] + embeds['TAC'] + embeds['GAC']) / 4
# embeds['NNC'] = (embeds['ACC'] + embeds['CTC'] + embeds['CCC'] + embeds['CGC'] + embeds['GTC'] + embeds['GCC'] + embeds['GGC'] + embeds['TCC'] + embeds['ATC'] + embeds['AAC'] + embeds['AGC'] + embeds['CAC'] + embeds['GAC'] + embeds['TTC'] + embeds['TAC'] + embeds['TGC']) / 16
# embeds['NCC'] = (embeds['ACC'] + embeds['CCC'] + embeds['TCC'] + embeds['GCC']) / 4
# embeds['NGC'] = (embeds['AGC'] + embeds['CGC'] + embeds['TGC'] + embeds['GGC']) / 4
# embeds['NCA'] = (embeds['ACA'] + embeds['CCA'] + embeds['TCA'] + embeds['GCA']) / 4
# embeds['NGA'] = (embeds['AGA'] + embeds['CGA'] + embeds['TGA'] + embeds['GGA']) / 4

print(len(list(embeds.keys())))
# print(embeds)
np.save('../../data/rb_prof_Naef/processed_proper/codons_embeds.npy', embeds) 

