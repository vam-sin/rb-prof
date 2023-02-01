import pandas as pd 
from Bio import SeqIO 
import numpy as np 
import pickle as pkl

records_ds = []
sequences_ds = []

with open('../data/rb_prof_Naef/processed_proper/seq_annot_raw/uniprot_not_matches_withX_withSmallest.fasta') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        records_ds.append(record.description)
        sequences_ds.append(str(record.seq))

records_uniprot = {'ENSMUST00000082396.1': 'P03893'}

desc = []
uniprot_id_from_mm = []
sequence = []

with open('../mouse_prots/uniprot_mus_musculus.fasta') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        desc.append(record.description)
        uniprot_id_from_mm.append(record.description.split('|')[1])
        sequence.append(str(record.seq))

print(len(desc), len(sequence))

ds_matches = pd.read_csv('../mouse_prots/fin_resultDB.m8', sep = '\t', header=None)
ds_matches.columns = ["query", "target", "perc_id", "aln_len", "mismatches", "gap_open", "query_start", "query_end", "target_start", "target_end", "e-val", "bit-score"]
# print(ds_matches)

# f = open("../data/rb_prof_Naef/processed_proper/seq_annot_raw/uniprot_not_matches_withX_withSmallest.fasta", "w")

for i in range(len(sequences_ds)):
    print(i, records_ds[i], len(sequences_ds))
    if sequences_ds[i][len(sequences_ds[i])-1] == 'X':
        edit_seq = sequences_ds[i][0:len(sequences_ds[i])-1]
        # print(edit_seq, sequences_ds[i])
    else:
        edit_seq = sequences_ds[i]
    
    if edit_seq in sequence:
        index_ = sequence.index(edit_seq)
        records_uniprot[records_ds[i]] = desc[index_]
    else:
        ds_matches_small = ds_matches[ds_matches["query"].isin([records_ds[i]])]
        aln_len_lis = list(ds_matches_small["aln_len"])
        if len(edit_seq) in aln_len_lis:
            ds_matches_small = ds_matches_small[ds_matches_small["aln_len"].isin([len(edit_seq)])]
            # print(ds_matches_small)
            target_id = list(ds_matches_small["target"])[0]
            # index_ = sequence.index(edit_seq)
            # print(index_, i, sequence[index_], sequences_ds[i])
            records_uniprot[records_ds[i]] = target_id
        else:
            # print(ds_matches_small)
            ds_matches_small.sort_values(by=['perc_id'])
            ds_matches_small.sort_values(by=['aln_len'], ascending=False)
            # print(ds_matches_small)
            if len(list(ds_matches_small["target"])) != 0:
                match_id = list(ds_matches_small["target"])[0]
                index_match = uniprot_id_from_mm.index(match_id)
                if len(sequence[index_match]) == len(edit_seq):
                    records_uniprot[records_ds[i]] = match_id
                    print(match_id, records_ds[i])
                else:
                    # print(ds_matches_small)
                    records_uniprot[records_ds[i]] = match_id
                    print("Not same length", len(sequence[index_match]), len(edit_seq))
                    # f.write('>' + records_ds[i] + '\n')
                    # f.write(sequences_ds[i] + '\n')
                    # break
            else:
                print("No matches")
                # break
                # f.write('>' + records_ds[i] + '\n')
                # f.write(sequences_ds[i] + '\n')
                # print(records_ds[i], ds_matches_small, len(edit_seq))
                # break
        
# f.close()
print(len(records_uniprot.keys()))

# with open('../data/rb_prof_Naef/processed_proper/seq_annot_raw/transcript_uniprot_match_withX_withSmallest.pkl', 'wb') as f:
#     pkl.dump(records_uniprot, f)

'''
ENSMUST00000208907.1
t: MERSSVKLGSRTPWRQVLFPFLLPLFCTGLSEQVRYSIPEEMAMGSVVGNLAEDLGLPVQDLLTRNLRVIAEKPYLSVNPENGNIVVSDRIDREFLCFQSPLCVLPLEIVAENPLNVFHVSVVIEDINDNPPRFLQNSIVLQINELAIPGTRFGLESAIDADVGLNSLQSYQLSLNEHFSLVVKDNTEGKDAPELVLEKPLDREKQSSQLLVLTAVDGGEPVLTGTAQIQIEVTDANDNPPVFSQSTYKVSLREDMAAGTSVLTVIATDQDEGVNAEVTYSFKSLGEDIRDKFILDHQSGEIKSKGPIDFETKRTYTMNIEAKDGGGMASECKVVVEILDENDNAPEVVFTSVSNSITEDAEPGTVVALFKTYDKDSEENGRVSCFVKETVPFRIESSASNYYKLVTDGILDREQTPEYNVTIIATDKGKPPLSSSTSVTLHVGDINDNAPVFHQTSYLIQVAENNPPGASIAQVSAFDPDLGSNGFISYSIIASDLEPKSLWSYVSVNQDSGVVFAQRAFDHEQLRSFQLTLQARDQGKPSLSANVSMRVLVGDRNDNAPRVLYPTLEPDGSALFDMVPRAAEPGYLVTKVVAVDADSGHNAWLSYHVLQASDPGLFSLGLRTGEVRTARALGDRDSARQRLLVAVRDGGQPPLSATATLHLIFADSLQEVLPDLRDEPLLSDSQSELQFYLVVALALVSVLFLFVVILAIVLRLRQSHGPAVSDYFQSGLCCKTRPEVSLNXGEGTLPYSYNLYVASNCQKTISQFLTLTPEMVPPRDLCTEASVAVSVAEENNKIVSDSIASNHQAPPNTDWRFSQAQRPGTSGSQNGDETGTWPNNQFDTEMLQAMILASASEAADGSSTLGGGAGTMGLSARYGPQFTLQHVPDYRQNVYIPGSNATLTNAAGKRDGKAPAGGNGNKKKSGKKEKKX
u: MERSSVKLGSRTPWRQVLFPFLLPLFCTGLSEQVRYSIPEEMAMGSVVGNLAEDLGLPVQDLLTRNLRVIAEKPYLSVNPENGNIVVSDRIDREFLCFQSPLCVLPLEIVAENPLNVFHVSVVIEDINDNPPRFLQNSIVLQINELAIPGTRFGLESAIDADVGLNSLQSYQLSLNEHFSLVVKDNTEGKDAPELVLEKPLDREKQSSQLLVLTAVDGGEPVLTGTAQIQIEVTDANDNPPVFSQSTYKVSLREDMAAGTSVLTVIATDQDEGVNAEVTYSFKSLGEDIRDKFILDHQSGEIKSKGPIDFETKRTYTMNIEAKDGGGMASECKVVVEILDENDNAPEVVFTSVSNSITEDAEPGTVVALFKTYDKDSEENGRVSCFVKETVPFRIESSASNYYKLVTDGILDREQTPEYNVTIIATDKGKPPLSSSTSVTLHVGDINDNAPVFHQTSYLIQVAENNPPGASIAQVSAFDPDLGSNGFISYSIIASDLEPKSLWSYVSVNQDSGVVFAQRAFDHEQLRSFQLTLQARDQGKPSLSANVSMRVLVGDRNDNAPRVLYPTLEPDGSALFDMVPRAAEPGYLVTKVVAVDADSGHNAWLSYHVLQASDPGLFSLGLRTGEVRTARALGDRDSARQRLLVAVRDGGQPPLSATATLHLIFADSLQEVLPDLRDEPLLSDSQSELQFYLVVALALVSVLFLFVVILAIVLRLRQSHGPAVSDYFQSGLCCKTRPEVSLNYSEGTLPYSYNLYVASNCQKTISQFLTLTPEMVPPRDLCTEASVAVSVAEENNKIVSDSIASNHQAPPNTDWRFSQAQRPGTSGSQNGDETGTWPNNQFDTEMLQAMILASASEAADGSSTLGGGAGTMGLSARYGPQFTLQHVPDYRQNVYIPGSNATLTNAAGKRDGKAPAGGNGNKKKSGKKEKK
X replaced with Y

ENSMUST00000082421.1
t: MTNIRKTHPLFKIINHSFIDLPAPSNISSXXNFGSLLGVCLIVQIITGLFLAIHYTSDTITAFSSVTHICRDVNYGXLIRYIHANGASIFFICLFLHVGRGLYYGSYTFIETXNIGVLLLFAVIATAFIGYVLPXGQISFXGATVITNLLSAIPYIGTTLVEXIXGGFSVDKATLTRFFAFHFILPFIIAALAIVHLLFLHETGSNNPTGLNSDADKIPFHPYYTIKDILGILIIFLILITLVLFFPDILGDPDNYIPANPLNTPPHIKPEXYFLFAYAILRSIPNKLGGVLALILSILILALIPFLHTSKQRSLIFRPITQILYXILVANLLILTXIGGQPVEHPFIIIGQLASISYFSIILILIPISGIIEDKILKLYP
u: MTNMRKTHPLFKIINHSFIDLPAPSNISSWWNFGSLLGVCLMVQIITGLFLAMHYTSDTMTAFSSVTHICRDVNYGWLIRYMHANGASMFFICLFLHVGRGLYYGSYTFMETWNIGVLLLFAVMATAFMGYVLPWGQMSFWGATVITNLLSAIPYIGTTLVEWIWGGFSVDKATLTRFFAFHFILPFIIAALAIVHLLFLHETGSNNPTGLNSDADKIPFHPYYTIKDILGILIMFLILMTLVLFFPDMLGDPDNYMPANPLNTPPHIKPEWYFLFAYAILRSIPNKLGGVLALILSILILALMPFLHTSKQRSLMFRPITQILYWILVANLLILTWIGGQPVEHPFIIIGQLASISYFSIILILMPISGIIEDKMLKLYP
X replaced with W
'''

'''
could extract 15602 exact uniprot matches for the proteins (out of 17784)
the other 2182 can be be salvaged because of diff transcript annots / uniprot links

559 can be matched when the X's 
491 can be matched with the best hit in MM with the same length (same sequences but are translated a little differently, this is what is on the website, check with Maria once)
Ex: F6YUG5 ENSMUST00000138676.1
'''

'''
make AF2 predictions for the leftover 1132
'''
