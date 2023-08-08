import pandas as pd 
from Bio import SeqIO

seq = []
desc = []

with open("../../data/rb_prof_Naef/processed_proper/seq_annot_final/translated_proteins.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq.append(str(record.seq))
        desc.append(record.description)

df = pd.DataFrame(list(zip(desc, seq)),
               columns =['Record', 'Sequence'])

print(df)
df.to_csv('../../data/rb_prof_Naef/processed_proper/seq_annot_final/Prot_CTRL_Dataset.csv')