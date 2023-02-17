# libraries
import pickle as pkl

with open('../data/rb_prof_Naef/processed_data/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_final.pkl', 'rb') as f:
    dict_Tr_Seq_Counts = pkl.load(f)

keys_list = list(dict_Tr_Seq_Counts.keys())
print(len(keys_list))

print(max(dict_Tr_Seq_Counts[keys_list[11]][1])*1e+2)
print(dict_Tr_Seq_Counts[keys_list[11]][0])
