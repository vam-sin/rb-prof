'''
this file removes those sequences that do not have any annotations
'''
# libraries
import pickle as pkl 
import numpy as np 

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/GLM/train_CTRL.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

print(len(dict_Tr_Seq.keys()))
dict_keys = list(dict_Tr_Seq.keys())
# print(dict_keys)

# for i in range(len(dict_keys)):
#     print(dict_Tr_Seq[dict_keys[i]])

def check_zeros(vals):
    vals = np.asarray(vals)
    return all(vals == 0.0)

for i in range(len(dict_keys)):
    if check_zeros(dict_Tr_Seq[dict_keys[i]][1]):
        del dict_Tr_Seq[dict_keys[i]]
    # else:
    #     print(dict_Tr_Seq[dict_keys[i]][1])
    #     # break

print(len(dict_Tr_Seq.keys()))

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/GLM/train_CTRL_proc.pkl', 'wb') as f:
    pkl.dump(dict_Tr_Seq, f)


'''
total of 66760 sequences, 17784 sequences have non-zero counts arrays
'''