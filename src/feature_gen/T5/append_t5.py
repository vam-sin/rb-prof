import numpy as np
import pandas as pd 
import pickle

filename = '/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/embeddings/' + 'T5_' + '0.0' + '.npz'
pb_arr = np.load(filename, allow_pickle=True)['arr_0']

for i in range(1, 100):
	print(i, pb_arr.shape)
	try:
		filename = '/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/embeddings/' + 'T5_' + str(i) + '.0' + '.npz'
		arr = np.load(filename, allow_pickle=True)['arr_0']
		pb_arr = np.append(pb_arr, arr, axis = 0)
	except:
		pass

print(pb_arr.shape)
# np.savez_compressed('/mnt/scratch/lts2/nallapar/rb-prof/CTRL_Embeddings_Full_Res_T5.npz', pb_arr)

ds = pd.read_csv('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/protein_sequences.csv')

labels = list(ds["transcript_label"])

tr_t5 = {}

for i in range(len(pb_arr)):
    tr_t5[labels[i]] = pb_arr[i]

with open('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/Embeddings_Full_Res_T5.pkl', 'wb') as f:
    pickle.dump(tr_t5, f)

