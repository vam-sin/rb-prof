'''
this script removes those sequences that do not have any annotations
'''
# libraries
import pickle as pkl 
import numpy as np 

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/seq_annot_raw/codon_annot_VAL.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

print(len(dict_Tr_Seq.keys()))
dict_keys = list(dict_Tr_Seq.keys())

def check_zeros(vals):
    vals = np.asarray(vals)
    return all(vals == 0.0)

def check_thresh(vals, num_counts, perc_counts):
    vals = np.asarray(vals)
    num_non_zero = np.count_nonzero(vals)
    if num_non_zero >= num_counts and (num_non_zero/len(vals)) >= perc_counts:
        return False
    else:
        return True

for i in range(len(dict_keys)):
    if check_zeros(dict_Tr_Seq[dict_keys[i]][1]) or check_thresh(dict_Tr_Seq[dict_keys[i]][1], 20, 0.2):
        del dict_Tr_Seq[dict_keys[i]]

print(len(dict_Tr_Seq.keys()))

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh02/seq_annot_final/fin_VAL.pkl', 'wb') as f:
    pkl.dump(dict_Tr_Seq, f)

'''Full
CTRL: 17785
ILE: 16789
LEU_ILE: 15854
LEU_ILE_VAL: 13168
LEU: 16528
VAL: 12857
'''

'''0.6/20
CTRL: 3293
ILE: 1759
LEU_ILE: 926
LEU_ILE_VAL: 83 
LEU: 1634
VAL: 57
Total: 7752
'''

'''0.4/20
CTRL: 5747
ILE: 3934
LEU_ILE: 2535
LEU_ILE_VAL: 340
LEU: 3572
VAL: 266
Total: 16,394
'''


'''0.2/20
CTRL: 8269
ILE: 7103
LEU_ILE: 5598
LEU_ILE_VAL: 1321
LEU: 6594
VAL: 1097
'''