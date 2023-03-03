import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from os.path import isfile, join 
from os import listdir
import pickle as pkl

# def num_non_zero_counts(y_counts):
#     counts = 0
#     for x in range(len(y_counts)):
#         if y_counts[x] != 0.0:
#             counts += 1

#     return counts, counts/len(y_counts)

# # import the dataset
# mypath = '/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon/'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# num_count_arr = []
# perc_counts_arr = []

# values_dict = {}

# for i in range(len(onlyfiles)):
#     print(i, len(onlyfiles))
#     key = onlyfiles[i]
#     filename_ = '/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon/' + key
#     arr = np.load(filename_, allow_pickle=True)['arr_0'].item()
#     y = np.absolute(arr['counts'])
#     num_counts, perc_counts = num_non_zero_counts(y)
#     print(num_counts, perc_counts)

#     num_count_arr.append(num_counts)
#     perc_counts_arr.append(perc_counts)
#     values_dict[key] = {'num_counts': num_counts, 'perc_counts': perc_counts}

# with open('ds_counts.pkl', 'wb') as f:
#     pkl.dump(values_dict, f)

# plt.hist(num_count_arr, density=True, bins=10)  # density=False would make counts
# plt.ylabel('Num Counts')
# plt.xlabel('Counts')
# plt.save_fig("fig1.png", format="png")

with open('ds_counts.pkl', 'rb') as f:
    count_ds = pkl.load(f)

keys_lis = list(count_ds.keys())
keys_proc = []

num_counts = []
perc_counts = []

for i in range(len(keys_lis)):
    if (count_ds[keys_lis[i]]['num_counts']) > 20 and (count_ds[keys_lis[i]]['perc_counts']) > 0.8:
        keys_proc.append(keys_lis[i])

print(len(keys_lis), len(keys_proc))

with open('keys_proc_20c_80p.pkl', 'wb') as f:
    pkl.dump(keys_proc, f)

'''
20, 0.8: 1193 (0.067)
20, 0.6: 3284 (0.184)
20, 0.4: 5739 (0.322)
20, 0.2: 8262 (0.464)
'''