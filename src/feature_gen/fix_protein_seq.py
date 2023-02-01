# libraries
import pickle as pkl 
import numpy as np
import random
import pandas as pd 
from Bio import SeqIO

# import data 
with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/ensembl_Tr_Seq_CTRL_merged_finalNonEmpty.pkl', 'rb') as f:
    dict_seqCounts = pkl.load(f)

keys_list = list(dict_seqCounts.keys())

seq = []
desc_uni = []

with open("../../mouse_prots/uniprot_mus_musculus.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq.append(str(record.seq))
        desc_uni.append(record.description.split('|')[1])

uniprot_seq = dict(zip(desc_uni, seq))

# print(uniprot_seq)

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_wProt.pkl', 'rb') as f:
    old_dict_seqCounts = pkl.load(f)

can_use_uniprots = []

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/AF2_files/transcript_uniprot_match.pkl', 'rb') as f:
    file_1 = pkl.load(f)

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/AF2_files/transcript_uniprot_match_withX.pkl', 'rb') as f:
    file_2 = pkl.load(f)

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/AF2_files/transcript_uniprot_match_withX_withSmallest.pkl', 'rb') as f:
    file_3 = pkl.load(f)

# print(file_1)

full_file = {**file_1, **file_2, **file_3}
can_use_tr = list(full_file.keys())

for i in range(len(can_use_tr)):
    print(i, full_file[can_use_tr[i]].split('|'))
    sp = full_file[can_use_tr[i]].split('|')
    if len(sp) > 1:
        uniprot_id = sp[1]
    else: 
        uniprot_id = sp[0]
    dict_seqCounts[can_use_tr[i]].append(uniprot_seq[uniprot_id])
    # print(dict_seqCounts[can_use_tr[i]])

need_translated_pro_tr = []
can_use_tr = set(can_use_tr)

for x in keys_list:
    if x not in can_use_tr:
        need_translated_pro_tr.append(x)

for i in range(len(need_translated_pro_tr)):
    print(i, len(need_translated_pro_tr))
    dict_seqCounts[need_translated_pro_tr[i]].append(old_dict_seqCounts[need_translated_pro_tr[i]][3])
    # print(dict_seqCounts[need_translated_pro_tr[i]])

# sanity check
count = 0 
for i in range(len(keys_list)):
    if len(dict_seqCounts[keys_list[i]]) == 4:
        count += 1

print(count, len(keys_list))

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/ensembl_Tr_Seq_CTRL_merged_wProt_fixed.pkl', 'wb') as f:
    pkl.dump(dict_seqCounts, f)