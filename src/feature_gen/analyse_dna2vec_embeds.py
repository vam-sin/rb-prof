import numpy as np 
import umap
import matplotlib.pyplot as plt 

filename = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/codons_embeds.npy'
# filename = 'DNABERT_Codon_Embeds_NAv.npy'

codon_embeds = np.load(filename, allow_pickle=True).item()
# for i in codon_embeds:
#     print(i)
# print(codon_embeds)
keys = list(codon_embeds.keys())
embeds = []
c = []
for i in range(len(keys)):
    embeds.append(codon_embeds[keys[i]])

reducer = umap.UMAP()

embeds_2d = reducer.fit_transform(embeds)

for i in range(len(keys)):
    col = '#be2edd'
    if keys[i] in ['TTT', 'TTC', 'TTA',' TTG']:
        col = '#f6e58d'
    if keys[i] in ['TCT', 'TCC', 'TCA', 'TCG']:
        col = '#ffbe76'
    if keys[i] in ['TAT', 'TAC']:
        col = '#f0932b'
    if keys[i] in ['TAA', 'TAG', 'TGA']:
        col = '#130f40'
    if keys[i] in ['TGT', 'TGC',]:
        col = '#ff7979'
    if keys[i] in ['CTT', 'CTC', 'CTA', 'CTG']:
        col = '#e056fd'
    if keys[i] in ['CCT', 'CCC', 'CCA', 'CCG']:
        col = '#22a6b3'
    if keys[i] in ['CGT', 'CGC', 'CGA', 'CGG']:
        col = '#c7ecee'
    if keys[i] in ['ATT', 'ATC', 'ATA']:
        col = '#bdc3c7'
    if keys[i] in ['ACT', 'ACC', 'ACA', 'ACG']:
        col = '#16a085'
    if keys[i] in ['AAT', 'AAC']:
        col = '#c0392b'
    if keys[i] in ['AAA', 'AAG']:
        col = '#2c2c54'
    if keys[i] in ['AGT', 'AGC']:
        col = '#227093'
    if keys[i] in ['AGA', 'AGG']:
        col = '#474787'
    if keys[i] in ['GTT', 'GTC', 'GTA', 'GTG']:
        col = '#34ace0'
    if keys[i] in ['GCT', 'GCC', 'GCA', 'GCG']:
        col = '#ff5252'
    if keys[i] in ['GAT', 'GAC']:
        col = '#FD7272'
    if keys[i] in ['GAA', 'GAG']:
        col = '#9AECDB'
    if keys[i] in ['GGT', 'GGC', 'GGA', 'GGG']:
        col = '#D6A2E8'
    if 'N' in keys[i]:
        col = '#6ab04c'

    plt.scatter(
    embeds_2d[i][0],
    embeds_2d[i][1],c=col)
plt.title('UMAP projection of DNA2Vec Codon Embeds', fontsize=24)
plt.show()
plt.savefig('umap_DNA2Vec.png')
