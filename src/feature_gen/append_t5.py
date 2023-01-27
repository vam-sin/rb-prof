import numpy as np
import pandas as pd 
import pickle

filename = '/mnt/scratch/lts2/nallapar/rb-prof/CTRL_T5_Embeddings/' + 'T5_' + '0.0' + '.npz'
pb_arr = np.load(filename, allow_pickle=True)['arr_0']

for i in range(1, 100):
	print(i, pb_arr.shape)
	try:
		filename = '/mnt/scratch/lts2/nallapar/rb-prof/CTRL_T5_Embeddings/' + 'T5_' + str(i) + '.0' + '.npz'
		arr = np.load(filename, allow_pickle=True)['arr_0']
		pb_arr = np.append(pb_arr, arr, axis = 0)
	except:
		pass

print(pb_arr.shape)
# np.savez_compressed('/mnt/scratch/lts2/nallapar/rb-prof/CTRL_Embeddings_Full_Res_T5.npz', pb_arr)

ds = pd.read_csv('../../data/rb_prof_Naef/processed_proper/seq_annot_final/Prot_CTRL_Dataset.csv')

labels = list(ds["Record"])

tr_t5 = {}

for i in range(len(pb_arr)):
    tr_t5[labels[i]] = pb_arr[i]

with open('/mnt/scratch/lts2/nallapar/rb-prof/CTRL_Embeddings_Full_Res_T5.pkl', 'wb') as f:
    pickle.dump(tr_t5, f)

