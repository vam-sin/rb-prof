import pandas as pd 
import pickle as pkl
import itertools

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
print(id_to_codon)
codon_to_id = {v:k for k,v in id_to_codon.items()}

# def fucntion sequence to codon ids
def sequence2codonids(seq):
    codon_ids = []
    for i in range(0, len(seq), 3):
        codon = seq[i:i+3]
        if len(codon) == 3:
            codon_ids.append(codon_to_id[codon])

    return codon_ids

## PKL

df = pd.read_pickle('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/raw/liver_na_1000.pkl')

print(df)

# remove transcripts with N in sequence
df = df[df['sequence'].str.contains('N') == False]

print(df)

codon_seqs = []
sequences = list(df['sequence'])
genes = list(df['gene'])
transcripts = list(df['transcript'])
perc_non_zero_annots = []
norm_counts = list(df['norm_counts'])
codon_idx = list(df["codon_idx"])
annot_seqs = []

for i in range(len(sequences)):
    seq = sequences[i]
    seq = sequence2codonids(seq)
    codon_seqs.append(seq)
    codon_idx_sample = codon_idx[i]
    # convert to list of int
    # codon_idx_sample = [int(i) for i in codon_idx_sample[1:-1].split(',')]
    annot_seq_sample = []
    norm_counts_sample = norm_counts[i]
    
    for j in range(len(seq)):
        if j in codon_idx_sample:
            annot_seq_sample.append(norm_counts_sample[codon_idx_sample.index(j)])
        else:
            annot_seq_sample.append(0.0)
    annot_seqs.append(annot_seq_sample)

    # calculate percentage of non-zero annotations
    perc_non_zero_annots.append(sum([1 for i in annot_seq_sample if i != 0.0])/len(annot_seq_sample))

final_df = pd.DataFrame(list(zip(genes, transcripts, codon_seqs, annot_seqs, perc_non_zero_annots)), columns = ['gene', 'transcript', 'codon_sequence', 'annotations', 'perc_non_zero_annots'])

print(final_df)

# save dataframe
final_df.to_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/liver_na_84.csv', index=False)
