from os import listdir
from os.path import isfile, join
import numpy as np 
import pickle as pkl

filename = 'DNABERT_Codon_Embeds_NAv.npy'
DNABERT_codon_embeds = np.load(filename, allow_pickle=True).item()
DNABERT_codon_embeds['---'] = np.zeros((768,))
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

output_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon_DNABERT/'

RNA_dict = {'A': [1,0,0,0,0], 'T': [0,1,0,0,0], 'G': [0,0,1,0,0], 'C': [0,0,0,1,0], 'N': [0,0,0,0,1]}

def inv_nt(nt_oh):
    if nt_oh[0] == 1:
        return 'A'
    if nt_oh[1] == 1:
        return 'T'
    if nt_oh[2] == 1:
        return 'G'
    if nt_oh[3] == 1:
        return 'C'
    if nt_oh[4] == 1:
        return 'N'
    else:
        return '-'

def nt2codon(nt_vec):
    codon = ''
    # print(nt_vec)
    # print(inv_nt(nt_vec[:5]))
    codon += inv_nt(nt_vec[:5])
    codon += inv_nt(nt_vec[5:10])
    codon += inv_nt(nt_vec[10:])
    # print(codon)
    return codon

for i in range(len(onlyfiles)):
    print(i, len(onlyfiles))
    filename_sample = mypath + onlyfiles[i]
    arr = np.load(filename_sample, allow_pickle=True)['arr_0'].item()
    # print(arr.keys())
    ft_vec = arr['feature_vec']
    nt_list = ft_vec[:,:15]
    codon_embed_list = []
    # print(nt_list.shape)
    for x in range(len(nt_list)):
        codon = nt2codon(nt_list[x])
        codon_embed = DNABERT_codon_embeds[codon]
        codon_embed_list.append(codon_embed)
    codon_embed_list = np.asarray(codon_embed_list)
    t5_lrs = ft_vec[:,115:]
    # print(nt_list.shape, codon_embed_list.shape, t5_lrs.shape)
    full_ft_vec = np.concatenate((nt_list, codon_embed_list, t5_lrs), axis=1)
    # print(full_ft_vec.shape)
    arr.pop('feature_vec', None)
    # print(arr.keys())
    arr['feature_vec_DNABERT'] = full_ft_vec
    # print(arr.keys())
    out_ft_filename = output_path + str(onlyfiles[i]) + '.npz'
    np.savez_compressed(out_ft_filename, arr)

    # break
