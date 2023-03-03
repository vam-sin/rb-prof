import pickle as pkl
from sklearn.model_selection import train_test_split
import random 
import numpy as np

# reproducibility
random.seed(0)
np.random.seed(0)

with open('processed_keys/keys_proc_20c_60p.pkl', 'rb') as f:
    onlyfiles = pkl.load(f)

transcripts_ids = [x.replace('.npz','') for x in onlyfiles]

print(transcripts_ids[0])

with open('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_finalNonEmpty.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

gene_ids_list = []

for i in range(len(transcripts_ids)):
    gene_id = dict_Tr_Seq[transcripts_ids[i]][2]
    gene_ids_list.append(gene_id)

gene_ids_list_set = list(set(gene_ids_list))
print(len(transcripts_ids), len(gene_ids_list))

gene_train, gene_test = train_test_split(gene_ids_list_set, test_size=0.2, random_state=42, shuffle=True)
gene_train, gene_val = train_test_split(gene_train, test_size=0.25, random_state=42, shuffle=True)

tr_train = []
tr_val = []
tr_test = []

for i in range(len(gene_train)):
    index_ = gene_ids_list.index(gene_train[i])
    tr_train.append(transcripts_ids[index_])

for i in range(len(gene_val)):
    index_ = gene_ids_list.index(gene_val[i])
    tr_val.append(transcripts_ids[index_])

for i in range(len(gene_test)):
    index_ = gene_ids_list.index(gene_test[i])
    tr_test.append(transcripts_ids[index_])

print(tr_train[0])
print(len(tr_train), len(tr_val), len(tr_test))

with open('data_split/train_20c_60p.pkl', 'wb') as f:
    pkl.dump(tr_train, f)

with open('data_split/val_20c_60p.pkl', 'wb') as f:
    pkl.dump(tr_val, f)

with open('data_split/test_20c_60p.pkl', 'wb') as f:
    pkl.dump(tr_test, f)