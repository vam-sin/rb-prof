import pickle as pkl 
import requests

with open('../data/rb_prof_Naef/processed_proper/seq_annot_raw/transcript_uniprot_match_withX_withSmallest.pkl', 'rb') as f:
    tr_uniprot = pkl.load(f)

keys_lis = list(tr_uniprot.keys())

for i in range(len(keys_lis)):
    print(i)
    print(tr_uniprot[keys_lis[i]])
    uniprot_id = tr_uniprot[keys_lis[i]]
    print(uniprot_id)
    url = 'https://alphafold.ebi.ac.uk/files/AF-' + str(uniprot_id) + '-F1-model_v4.pdb'
    response = requests.get(url)
    download_file = "/mnt/scratch/lts2/nallapar/AF2/AF2_Downloads/" + str(keys_lis[i]) + ".pdb"
    open(download_file, "wb").write(response.content)


    