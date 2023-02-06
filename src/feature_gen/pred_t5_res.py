# libraries
import numpy as np
from bio_embeddings.embed import ProtTransT5BFDEmbedder
import pandas as pd 

print("Starting")
embedder = ProtTransT5BFDEmbedder()
print("Loaded")
ds = pd.read_csv('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/T5_files/Prot_CTRL_Dataset.csv')

sequences_Example = list(ds["Sequence"])
# num_seq = len(sequences_Example)
num_seq = 7500

i = 0
length = 500
while i < num_seq:
	print("Doing", i, num_seq)
	start = i 
	end = i + length

	sequences = sequences_Example[start:end]

	embeddings = []
	for seq in sequences:
		if len(seq) > 10000:
			big_seq_len = len(seq)
			x = 0
			diff = 5000
			embed_arr = []
			while x < big_seq_len:
				embed_arr.append(np.asarray(embedder.embed(seq[x:min(big_seq_len, x+diff)])))
				x += diff
			append_arr = np.concatenate((embed_arr), axis=0)
			print(append_arr.shape)
			embeddings.append(np.asarray(append_arr))
		else:
			embeddings.append(np.asarray(embedder.embed(seq)))

	s_no = start / length
	filename = '/mnt/scratch/lts2/nallapar/rb-prof/CTRL_T5_Embeddings/' + 'T5_' + str(s_no) + '.npz'
	embeddings = np.asarray(embeddings)
	# print(embeddings.shape)
	np.savez_compressed(filename, embeddings)
	i += length

'''

'''
