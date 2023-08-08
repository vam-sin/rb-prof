import pandas as pd 
from Bio import SeqIO 
import numpy as np 
import pickle as pkl

transcript = []
gene = []
transcript_gene = []
sequence = []
counts_array = []
seq_counts = []

with open('../data/rb_prof_Naef/AA_depr/ensembl.cds.fa') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        s = record.description.split(' ')
        if str(record.seq) not in sequence:
            sequence.append(str(record.seq))
            tr_str = s[0]
            transcript.append(tr_str)
            counts_array.append(np.zeros((len(str(record.seq)),), dtype=int))
            seq_counts_el = [str(record.seq), np.zeros((len(str(record.seq)),), dtype=int)]
            seq_counts.append(seq_counts_el)
            for x in s:
                if 'gene:' in x:
                    gene_str = x.replace('gene:','')
                    gene.append(gene_str)
                    break

            str_tr_gen = tr_str + '_' + gene_str
            transcript_gene.append(str_tr_gen)

print(len(gene), len(transcript), len(transcript_gene), len(sequence))
print(len(list(set(gene))), len(list(set(transcript))), len(list(set(transcript_gene))), len(list(set(sequence))))

'''
66760 unique transcripts, 22950 unique genes, (66760 unique transcripts_genes), 56823 unique sequences

there are 56823 unique sequences: 22408 unique genes + 56823 unique transcripts

3 CTRL
3 ILE Deprivation
3 LEU Deprivation
3 LEU_ILE Deprivation
1 LEU_ILE_VAL Deprivation
1 VAL Deprivation

Total: 6 unique types of sets of counts (possibly 6*66760 == 400,560)
'''