# libraries
import pickle as pkl 
import multiprocessing as mp

withRNA = {}

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_RNA-SS_10k.pkl', 'rb') as f:
    withRNA_1 = pkl.load(f)
    f.close()

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_RNA-SS_A10k.pkl', 'rb') as f:
    withRNA_2 = pkl.load(f)
    f.close()

print(len(list(withRNA_1.keys())))
print(len(list(withRNA_2.keys())))

# merge both dicts
withRNA_full = {}

keys_list_1 = list(withRNA_1.keys())
keys_list_2 = list(withRNA_2.keys())

for i in range(len(keys_list_1)):
    withRNA_full[keys_list_1[i]] = withRNA_1[keys_list_1[i]]

for i in range(len(keys_list_2)):
    withRNA_full[keys_list_2[i]] = withRNA_2[keys_list_2[i]]

print(len(list(withRNA_full.keys())))

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_nonEmpty_wProtSeq.pkl', 'rb') as f:
    data = pkl.load(f)
    f.close()

# print(withRNA_full)
withRNA_full_keys = list(withRNA_full.keys())

for i in range(len(withRNA_full_keys)):
    print(i)
    rna_ss = withRNA_full[withRNA_full_keys[i]]
    # print(rna_ss)
    data[withRNA_full_keys[i]].append(rna_ss)

print(data[withRNA_full_keys[0]])

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_nonEMpty_wProtSeq_wRNA-SS.pkl', 'wb') as f:
    pkl.dump(data, f)

print("Finished")