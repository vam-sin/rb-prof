'''
finds uniprot hits for the protins in the mouse proteome
'''
import pandas as pd 
from Bio import SeqIO 
import numpy as np 
import pickle as pkl

prot_recs = []
prot_seqs = []
prot_rec_uniprots = {}

prot_recs_pred = []
prot_seqs_pred = []

with open('../../../data/mouse_prots/protein_sequences.fasta') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        prot_recs.append(record.description)
        prot_seqs.append(str(record.seq))

mm_desc = []
mm_uniprot = []
mm_seqs = []

with open('../../../data/mouse_prots/uniprot_mus_musculus_S100.fasta') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        mm_desc.append(record.description)
        mm_uniprot.append(record.description.split('|')[1])
        mm_seqs.append(str(record.seq))

# print(len(desc), len(sequence))

# print(desc[0], uniprot_id_from_mm[0], sequence[0])

# for i in range(len(prot_recs)):
#     print(i, len(prot_recs))
#     if prot_seqs[i] in mm_seqs:
#         prot_rec_uniprots[prot_recs[i]] = mm_uniprot[mm_seqs.index(prot_seqs[i])]
#         # print(prot_recs[i], mm_uniprot[mm_seqs.index(prot_seqs[i])], prot_seqs[i], mm_seqs[mm_seqs.index(prot_seqs[i])])
#     # print(prot_rec_uniprots)

# keys = list(prot_rec_uniprots.keys())

# print(len(keys), len(prot_recs))
# print(len(keys))

# with open('uniprot_mapped_proteins.pkl', 'wb') as f:
#     pkl.dump(prot_rec_uniprots, f)

with open('uniprot_mapped_proteins.pkl', 'rb') as f:
    data = pkl.load(f)

keys = list(set(list(data.keys())))
print("Number of Unique Transcripts:", len(list(set(prot_recs))))
print("Number of Unique Transcripts with Uniprot ID:", len(keys))
prot_recs = list(set(prot_recs))
# print(len(keys), len(prot_recs))
count = 0
count_2 = 0
for i in range(len(prot_recs)):
    if prot_recs[i] not in keys:
        count += 1
        prot_recs_pred.append(prot_recs[i])
        prot_seqs_pred.append(prot_seqs[i])
    else:
        count_2 += 1
        # print(prot_recs[i], data[prot_recs[i]])

# make fasta files for the proteins that are not in uniprot
for i in range(len(prot_recs_pred)):
    f = open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/AF2/alphafold-2.2.3/fasta_files/' + str(prot_recs_pred[i]) + '.fasta', 'w')
    f.write('>' + str(prot_recs_pred[i]) + '\n')
    f.write(prot_seqs_pred[i] + '\n')
    f.close()

print(count, count_2)

'''
3119 sequences can be downloaded from uniprot directly
'''

