import pickle as pkl 
import pandas as pd 

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_nonEmpty_wProtSeq.pkl', 'rb') as f:
    data = pkl.load(f)

keys_list = list(data.keys())
prot_seq = []

for i in range(len(keys_list)):
    prot_seq.append(data[keys_list[i]][3])

print(len(keys_list), len(prot_seq))

df = pd.DataFrame(list(zip(keys_list, prot_seq)), columns = ['Record', 'Sequence'])

df.to_csv('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/Prot_CTRL_Dataset.csv')


