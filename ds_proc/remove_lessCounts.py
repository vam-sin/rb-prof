# libraries
import pandas as pd 
import numpy as np
import pickle as pkl

# load in the data
ds = pd.read_csv('../data/rb_prof_Naef/processed_data/merge_norm/merge_CTRL_RIBO_gnorm.csv')
ds.columns = ["index", "gene", "transcript", "position_A_site", "count", "count_GScale"]

transcripts = list(set(list(ds["transcript"])))

print(ds)

with open('../data/rb_prof_Naef/processed_data/ensembl_Tr_Seq.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

transcripts_keep = []
perc_list = []
thresh = 0.4

for i in range(len(transcripts)):
    ds_seg = ds[ds["transcript"].isin([transcripts[i]])]
    pos_A_seg = list(ds_seg["position_A_site"])
    seq_i = str(dict_Tr_Seq[transcripts[i]][0])
    perc = len(pos_A_seg) / (len(seq_i) -  2)
    perc_list.append(perc)
    # print(len(seq_i), seq_i)
    # print(i, len(transcripts), len(transcripts_keep), transcripts[i], len(seq_i), perc)
    print(i, len(transcripts_keep), len(transcripts))
    if perc >= thresh: # 40% of the codons are covered
        transcripts_keep.append(transcripts[i])

print(len(transcripts), len(transcripts_keep))


ds = ds[ds["transcript"].isin(transcripts_keep)]

ds.to_csv('../data/rb_prof_Naef/processed_data/merge_norm_proc/merge_CTRL_RIBO_gnorm_proc.csv')

# 40% cutoff for the sequence coverage

'''
LEU_ILE_VAL: 0/13168

'''