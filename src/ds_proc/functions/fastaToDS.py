'''
function to convert a fasta file to a pandas dataframe
'''
import pandas as pd 
from Bio import SeqIO

seq = []
desc = []

with open("../data/rb_prof_Naef/processed_data_full/seq_annot_final/translated_proteins_CTRL.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq.append(str(record.seq))
        desc.append(record.description)

df = pd.DataFrame(list(zip(desc, seq)),
               columns =['Record', 'Sequence'])

print(df)
df.to_csv('../data/rb_prof_Naef/processed_data_full/seq_annot_final/Prots_CTRL.csv')