'''
converts dataframe into a {seq:count_seq} format
'''
import pandas as pd 
from Bio import SeqIO 
import numpy as np 
import pickle as pkl

# transcript = []
# gene = []
# sequence = []
# counts_array = []
# seq_counts = []

# with open('../data/rb_prof_Naef/AA_depr/ensembl.cds.fa') as handle:
#     for record in SeqIO.parse(handle, "fasta"):
#         s = record.description.split(' ')
#         tr_str = s[0]
#         transcript.append(tr_str)
#         sequence.append(str(record.seq))
#         counts_array.append(np.zeros((len(str(record.seq)),), dtype=int))
#         seq_counts_el = [str(record.seq), np.zeros((len(str(record.seq)),), dtype=int)]
#         seq_counts.append(seq_counts_el)
#         for x in s:
#             if 'gene:' in x:
#                 gene_str = x.replace('gene:','')
#                 gene.append(gene_str)
#                 break

# print(len(transcript), len(gene), len(sequence), len(counts_array))
# print(transcript[10], gene[10], str(sequence[10]), counts_array[10], len(counts_array[10]), len(str(sequence[10])))

# dict_trGene_Seq = dict(zip(transcript, seq_counts))

# with open('../data/rb_prof_Naef/processed_data/ensembl_Tr_Seq.pkl', 'wb') as f:
#     pkl.dump(dict_trGene_Seq, f)

# # mapping from transcript_Gene -> Sequence
with open('../data/rb_prof_Naef/processed_data/ensembl_Tr_Seq.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

ds = pd.read_csv('../data/rb_prof_Naef/processed_proper/merge_gnorm/merge_gnorm_CTRL.csv')
ds.columns = ["index", "gene", "transcript", "position_A_site", "count", "count_GScale"]
# print(ds)
# print(dict_Tr_Seq)

transcript_list = list(set(list(ds["transcript"])))

for i in range(len(transcript_list)):
    print(i, len(transcript_list))
    ds_tr = ds[ds["transcript"].isin([transcript_list[i]])]
    if len(ds_tr.index) != 0:
        # print(ds_tr)
        pos_A_sample = list(ds_tr["position_A_site"])
        counts_sample = list(ds_tr["count_GScale"])
        gene_name = list(set(list(ds_tr["gene"])))[0]
        # print(ds_tr)
        dict_Tr_Seq[transcript_list[i]][1] = np.asarray(dict_Tr_Seq[transcript_list[i]][1], dtype=np.float64)
        dict_Tr_Seq[transcript_list[i]].append(str(gene_name))
        # print(gene_name)
        for j in range(len(pos_A_sample)):
            # print(pos_A_sample[j])
            try:
                # all the nts in the A site are tagged with the read count values
                dict_Tr_Seq[transcript_list[i]][1][pos_A_sample[j]] = counts_sample[j]
                dict_Tr_Seq[transcript_list[i]][1][pos_A_sample[j] + 1] = counts_sample[j]
                dict_Tr_Seq[transcript_list[i]][1][pos_A_sample[j] + 2] = counts_sample[j]
                # print(counts_sample[j], dict_Tr_Seq[transcript_list[i]][1][pos_A_sample[j]-1])
            except:
                print(i, transcript_list[i], len(transcript_list))
                print("ERROR: ", pos_A_sample[j], len(dict_Tr_Seq[transcript_list[i]][1]))

        # print(np.asarray(dict_Tr_Seq[transcript_list[i]][1]))

with open('../data/rb_prof_Naef/processed_proper/seq_annot_raw/ensembl_Tr_Seq_CTRL_merged.pkl', 'wb') as f:
    pkl.dump(dict_Tr_Seq, f)

'''
all the three nts in the codon of the A site are tagged with the read count values. 
0 1 2 3 4 5 6 7 8  9 10 11 12 13 14 15 16 -> 0 index
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 -> 1 index
A T G C A T T T C  C  G  G  A  G  A  G  A
            E E E  P  P  P  A  A  A
seq: 12, 3 (pos, counts)

'''

'''
The A site values are 1 indexed,
tagging the P site with the values of the counts because A site can be outside of the se
is the A site position 0/1 indexed
'''

'''ERRORS in A site position tagging (CTRL MERGE)
592 ENSMUST00000149485.1 2402
ERROR:  756
'''