import pickle as pkl 
import requests

with open('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_raw/dataset_ProcFeats_fixed_prot.pkl', 'rb') as f:
    data = pkl.load(f)

data_keys = list(data.keys())

tr_uni = {}

for i in range(len(data_keys)):
    if 'Uniprot' not in data[data_keys[i]].keys():
        tr_uni[data_keys[i]] = data[data_keys[i]]['Protein_Seq']

tr_uni_keys = list(tr_uni.keys())

for i in range(len(tr_uni_keys)):
    filename = '/mnt/scratch/lts2/nallapar/AF2/alphafold-2.2.3/fasta_files/' + str(tr_uni_keys[i]) + '.fasta'
    f = open(filename, 'w')
    f.write('>' + str(tr_uni_keys[i]) + '\n')
    edit_seq = tr_uni[tr_uni_keys[i]]
    len_es = len(edit_seq)
    if edit_seq[len_es-1] == 'X':
        edit_seq = edit_seq[:len_es-1]
    edit_seq = edit_seq.replace('X', 'W') # all the
    f.write(edit_seq)
    f.close()

# for i in range(len(tr_uni_keys)):
#     print(i)
#     print(tr_uni[tr_uni_keys[i]])
#     uniprot_id = tr_uni[tr_uni_keys[i]]
#     print(uniprot_id)
#     url = 'https://alphafold.ebi.ac.uk/files/AF-' + str(uniprot_id) + '-F1-model_v4.pdb'
#     response = requests.get(url)
#     download_file = "/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/" + str(tr_uni_keys[i]) + ".pdb"
#     open(download_file, "wb").write(response.content)


    