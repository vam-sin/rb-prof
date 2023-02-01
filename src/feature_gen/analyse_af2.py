import pickle as pkl 
from Bio import SeqIO

seq = []
desc = []

with open("query.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq.append(str(record.seq))
        desc.append(record.description)

print(len(seq[0]), desc)

with open('features.pkl', 'rb') as f:
    feats = pkl.load(f)

print(feats['aatype'])