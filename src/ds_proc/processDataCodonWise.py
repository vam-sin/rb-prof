'''
converts dataframe into a {seq:count_seq} format
puts features according to codons
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

codon_table = {
        'ATA':1, 'ATC':2, 'ATT':3, 'ATG':4,
        'ACA':5, 'ACC':6, 'ACG':7, 'ACT':8,
        'AAC':9, 'AAT':10, 'AAA':11, 'AAG':12,
        'AGC':13, 'AGT':14, 'AGA':15, 'AGG':16,                
        'CTA':17, 'CTC':18, 'CTG':19, 'CTT':20,
        'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24,
        'CAC':25, 'CAT':26, 'CAA':27, 'CAG':28,
        'CGA':29, 'CGC':30, 'CGG':31, 'CGT':32,
        'GTA':33, 'GTC':34, 'GTG':35, 'GTT':36,
        'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40,
        'GAC':41, 'GAT':42, 'GAA':43, 'GAG':44,
        'GGA':45, 'GGC':46, 'GGG':47, 'GGT':48,
        'TCA':49, 'TCC':50, 'TCG':51, 'TCT':52,
        'TTC':53, 'TTT':54, 'TTA':55, 'TTG':56,
        'TAC':57, 'TAT':58, 'TAA':59, 'TAG':60,
        'TGC':61, 'TGT':62, 'TGA':63, 'TGG':64, 'NNG': 66, 'NGG': 67, 'NNT': 68,
        'NTG': 69, 'NAC': 70, 'NNC': 71, 'NCC': 72,
        'NGC': 73, 'NCA': 74, 'NGA': 75, 'NNA': 76,
        'NAG': 77, 'NTC': 78, 'NAT': 79, 'NGT': 80,
        'NCG': 81, 'NTT': 82, 'NCT': 83, 'NAA': 84,
        'NTA': 85
    }

def codonify(transcript_seq):
    # converts the codon sequence into codons
    codon_seq = []
    for i in range(0, len(transcript_seq), 3):
        try:
            codon_seq.append(codon_table[transcript_seq[i:i+3]])
        except:
            print("FAIL:",transcript_seq[i:i+3])
    return codon_seq


# # mapping from transcript_Gene -> Sequence
with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/ensembl_Tr_Seq.pkl', 'rb') as f:
    dict_Tr_Seq = pkl.load(f)

ds = pd.read_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/merge_gnorm/merge_gnorm_CTRL.csv')
ds.columns = ["index", "gene", "transcript", "position_A_site", "count", "count_GScale"]
# print(ds)
# print(dict_Tr_Seq)

transcript_list = list(set(list(ds["transcript"])))

codon_ds = {}

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
        # print(len(dict_Tr_Seq[transcript_list[i]][1]))
        codon_annots = []
        for k in range(0, len(dict_Tr_Seq[transcript_list[i]][1]), 3):
            # print(k, len(dict_Tr_Seq[transcript_list[i]][1]))
            if k + 2 <= len(dict_Tr_Seq[transcript_list[i]][1])-1:
                codon_annots.append(dict_Tr_Seq[transcript_list[i]][1][k])
        # print(codon_annots, len(codon_annots))
        codon_seq = codonify(dict_Tr_Seq[transcript_list[i]][0])
        # print(dict_Tr_Seq[transcript_list[i]][0])
        value = [codon_seq, codon_annots, gene_name]
        assert_out = str(len(codon_seq)) + " " + str(len(codon_annots))
        # print(assert_out)
        assert len(codon_seq) == len(codon_annots), assert_out
        codon_ds[transcript_list[i]] = value
        # print(codon_ds)

    # break

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/seq_annot_raw/codon_annot_CTRL.pkl', 'wb') as f:
    pkl.dump(codon_ds, f)

'''
all the three nts in the codon of the A site are tagged with the read count values. 
0 1 2 3 4 5 6 7 8  9 10 11 12 13 14 15 16 -> 0 index
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 -> 1 index
A T G C A T T T C  C  G  G  A  G  A  G  A
            E E E  P  P  P  A  A  A
C1    C2    C3     C4       C5    C6
                            
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