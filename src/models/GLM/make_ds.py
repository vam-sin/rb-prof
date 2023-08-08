# libraries
import numpy as np
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import pickle as pkl
import random
import pandas as pd 

# reproducibility
random.seed(0)
np.random.seed(0)

# import data files
with open('../regression/processed_keys/keys_proc_20c_20p.pkl', 'rb') as f:
    onlyfiles = pkl.load(f)

only_files_tr = []
for i in range(len(onlyfiles)):
    only_files_tr.append(onlyfiles[i].replace('.npz',''))
print(only_files_tr[0])

tr_train, tr_test = train_test_split(only_files_tr, test_size=0.2, random_state=42, shuffle=True)
tr_train, tr_val = train_test_split(tr_train, test_size=0.25, random_state=42, shuffle=True)

print(f'Train Set: {len(tr_train):5d} || Validation Set: {len(tr_val):5d} || Test Set: {len(tr_test): 5d}')

# # mapping from transcript_Gene -> Sequence
with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/ensembl_Tr_Seq.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

ds = pd.read_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/merge_NoNorm/merge_NoNorm_CTRL.csv')
ds.columns = ["index", "gene", "transcript", "position_A_site", "counts"]
print(ds)

# train_ds
for i in range(len(tr_train)):
    print(i, len(tr_test))
    ds_tr = ds[ds["transcript"].isin([tr_test[i]])]
    if len(ds_tr.index) != 0:
        # print(ds_tr)
        pos_A_sample = list(ds_tr["position_A_site"])
        counts_sample = list(ds_tr["counts"])
        gene_name = list(set(list(ds_tr["gene"])))[0]
        # print(ds_tr)
        dict_Tr_Seq[tr_test[i]][1] = np.asarray(dict_Tr_Seq[tr_test[i]][1], dtype=np.float64)
        dict_Tr_Seq[tr_test[i]].append(str(gene_name))
        # print(gene_name)
        for j in range(len(pos_A_sample)):
            # print(pos_A_sample[j])
            try:
                # all the nts in the A site are tagged with the read count values
                dict_Tr_Seq[tr_test[i]][1][pos_A_sample[j]] = counts_sample[j]
                dict_Tr_Seq[tr_test[i]][1][pos_A_sample[j] + 1] = counts_sample[j]
                dict_Tr_Seq[tr_test[i]][1][pos_A_sample[j] + 2] = counts_sample[j]
                # print(counts_sample[j], dict_Tr_Seq[transcript_list[i]][1][pos_A_sample[j]-1])
            except:
                print(i, tr_test[i], len(tr_test))
                print("ERROR: ", pos_A_sample[j], len(dict_Tr_Seq[tr_test[i]][1]))

        # print(dict_Tr_Seq[tr_train[i]])

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/GLM/test_CTRL.pkl', 'wb') as f:
    pkl.dump(dict_Tr_Seq, f)