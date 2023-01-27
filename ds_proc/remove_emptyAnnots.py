'''
this file removes those sequences that do not have any annotations
'''
# libraries
import pickle as pkl 
import numpy as np 

with open('../data/rb_prof_Naef/processed_proper/seq_annot_raw/ensembl_Tr_Seq_CTRL_merged.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

print(len(dict_Tr_Seq.keys()))
dict_keys = list(dict_Tr_Seq.keys())

def check_zeros(vals):
    vals = np.asarray(vals)
    return all(vals == 0.0)

for i in range(len(dict_keys)):
    if check_zeros(dict_Tr_Seq[dict_keys[i]][1]):
        # print(dict_Tr_Seq[dict_keys[i]][1])
        del dict_Tr_Seq[dict_keys[i]]

print(len(dict_Tr_Seq.keys()))

with open('../data/rb_prof_Naef/processed_proper/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_finalNonEmpty.pkl', 'wb') as f:
    pkl.dump(dict_Tr_Seq, f)


'''
2402 sequences in CTRL MERGE FINAL PROCESSED
17785 sequences in FULL CTRL MERGE FINAL
'''