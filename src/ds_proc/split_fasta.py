import pandas as pd 
from Bio import SeqIO

seq = []
desc = []

with open("../data/rb_prof_Naef/processed_proper/seq_annot_raw/af_pred.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq.append(str(record.seq))
        desc.append(record.description)

for i in range(len(seq)):
    print(i)
    filename = '/mnt/scratch/lts2/nallapar/AF2/alphafold-2.2.3/fasta_files/' + desc[i] + '.fasta'
    f = open(filename, 'w')
    f.write('>' + desc[i] + '\n')
    f.write(seq[i])
    f.close()

