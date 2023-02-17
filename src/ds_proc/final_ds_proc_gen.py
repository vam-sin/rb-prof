# libraries
import numpy as np 
import pickle as pkl

with open('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_raw/dataset_ProcFeats_fixed_prot.pkl', 'rb') as f:
    data = pkl.load(f)
    f.close()

with open('/mnt/scratch/lts2/nallapar/rb-prof/CTRL_Embeddings_Full_Res_T5.pkl', 'rb') as f:
    t5_embeds = pkl.load(f)
    f.close()

keys_list = list(data.keys())

def process_t5(embed, len_nt):
    print(len_nt, len(embed))
    len_embed = len(embed)
    out_embed = []
    for j in range(len_embed):
        for k in range(3):
            out_embed.append(embed[j])
    leftover_len = len_nt - len(out_embed)
    zero_arr = np.zeros((1024))
    for j in range(leftover_len):
        out_embed.append(zero_arr)

    out_embed = np.asarray(out_embed)

    return out_embed

def codonify(nt_codon_ft_vec, t5_vec, lem_vec, count_vec):
    seq_len = len(nt_codon_ft_vec)
    codonified_vec_X = []
    codonified_vec_y = []
    for j in range(0, seq_len, 3):
        if j+2 < seq_len:
            one_codon_vec = np.asarray([])

            # add the three nts vectors
            one_codon_vec = np.concatenate((one_codon_vec, nt_codon_ft_vec[j][:5]))
            one_codon_vec = np.concatenate((one_codon_vec, nt_codon_ft_vec[j+1][:5]))
            # add the third nt vector and the respective codon feature
            one_codon_vec = np.concatenate((one_codon_vec, nt_codon_ft_vec[j+2]))

            # add the t5 AA vector
            one_codon_vec = np.concatenate((one_codon_vec, t5_vec[int(j/3)]))

            # add the RNA SS LEM
            one_codon_vec = np.concatenate((one_codon_vec, lem_vec[j]))
            one_codon_vec = np.concatenate((one_codon_vec, lem_vec[j+1]))
            one_codon_vec = np.concatenate((one_codon_vec, lem_vec[j+2]))

            # count vecs
            codonified_vec_y.append(count_vec[j])
        else:
            one_codon_vec = np.zeros((1235))
            codonified_vec_y.append(0.0)

        codonified_vec_X.append(one_codon_vec)
    
    codonified_vec_X = np.asarray(codonified_vec_X)
    codonified_vec_y = np.asarray(codonified_vec_y)

    print(codonified_vec_X.shape, codonified_vec_y.shape, seq_len)
    assert len(codonified_vec_X) == len(codonified_vec_y)

    return codonified_vec_X, codonified_vec_y

for i in range(4389, len(keys_list)):
    print(i, keys_list[i])
    key_val = keys_list[i]
    # print(data[key_val])
    # print(t5_embeds[key_val].shape)
    len_nt = len(data[key_val]['nt_codon_ft'])
    proc_t5_embed = process_t5(t5_embeds[key_val], len_nt)
    print(data[key_val]['nt_codon_ft'].shape, proc_t5_embed.shape, data[key_val]['RNA_SSG_LEM'].shape)
    feature_vector, codon_counts = codonify(data[key_val]['nt_codon_ft'], t5_embeds[key_val], data[key_val]['RNA_SSG_LEM'], data[key_val]['Counts'])
    # feature_vector = np.concatenate((data[key_val]['nt_codon_ft'], proc_t5_embed, data[key_val]['RNA_SSG_LEM']), axis=1)
    out = dict(zip(['feature_vec', 'counts'], [feature_vector, codon_counts]))
    # break
    filename = '/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon/' + str(keys_list[i]) + '.npz'
    np.savez_compressed(filename, out)
    # break

'''
ProcFeats_Final ('seq_annot_final/final_dataset') files has the following features:
- NT One-Hot
- Codon DNA2Vec 
- RNA SS LEM
- T5 embeds
'''

'''
final_dataset_codon: has codon by codon feature sets
'''