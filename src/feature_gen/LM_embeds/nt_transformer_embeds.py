from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from os import listdir
from os.path import isfile, join
import pickle as pkl
import numpy as np
import gc

# codon table
codon_table = {
        'ATA':1, 'ATC':2, 'ATT':3, 'ATG':4,
        'ACA':5, 'ACC':6, 'ACG':7, 'ACT':8,
        'AAC':9, 'AAT':10, 'AAA':11, 'AAG':12,
        'AGC':13, 'AGT':14, 'AGA':15, 'AGG':16,                
        'CTA':17, 'CTC':18, 'CTG':19, 'CTT':20,
        'CCA':21, 'CCC':22, 'CCG':23, 'CCT':24,
        'CAC':25, 'CAT':26, 'CAA':27, 'CAG':28,
        'CGA':29, 'CGC':30, 'CGG':31, 'CGT':32,
        'GTA':33, 'GTC':34, 'GTG':35, 'GTT':36,
        'GCA':37, 'GCC':38, 'GCG':39, 'GCT':40,
        'GAC':41, 'GAT':42, 'GAA':43, 'GAG':44,
        'GGA':45, 'GGC':46, 'GGG':47, 'GGT':48,
        'TCA':49, 'TCC':50, 'TCG':51, 'TCT':52,
        'TTC':53, 'TTT':54, 'TTA':55, 'TTG':56,
        'TAC':57, 'TAT':58, 'TAA':59, 'TAG':60,
        'TGC':61, 'TGT':62, 'TGA':63, 'TGG':64, 'NNG': 66, 'NGG': 67, 'NNT': 68,
        'NTG': 69, 'NAC': 70, 'NNC': 71, 'NCC': 72,
        'NGC': 73, 'NCA': 74, 'NGA': 75, 'NNA': 76,
        'NAG': 77, 'NTC': 78, 'NAT': 79, 'NGT': 80,
        'NCG': 81, 'NTT': 82, 'NCT': 83, 'NAA': 84,
        'NTA': 85
    }

# inverse codon table
inv_codon_table = {v: k for k, v in codon_table.items()}

def getRNASeq(codon_seq):
    rna_seq = []
    for i in range(len(codon_seq)):
        rna_seq.append(inv_codon_table[codon_seq[i]])

    return rna_seq

# Import the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

# train files
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
train_files = [mypath + '/' + f for f in onlyfiles]

# test files
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_files = [mypath + '/' + f for f in onlyfiles]

# val files
mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
val_files = [mypath + '/' + f for f in onlyfiles]

# split into chunks of 200 and then send to the model
# train files
for i in range(len(train_files)):
    print(i)
    with open(train_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    codon_seq = np.asarray(data_dict['sequence'])

    rna_seq = getRNASeq(codon_seq)
    print(len(rna_seq))
    tokens_ids = tokenizer.batch_encode_plus(rna_seq, return_tensors="pt")["input_ids"]

    # Compute the embeddings
    attention_mask = tokens_ids != tokenizer.pad_token_id
    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
    )

    # Compute sequences embeddings
    embeddings = torch_outs['hidden_states'][-1].detach().numpy()
    # remove the CLS tokens
    embeddings = embeddings[:, 1:, :]
    # take mean of the embeddings for each codon
    embeddings = embeddings.mean(axis=1)
    print(f"Embeddings shape: {embeddings.shape}")
    # print(f"Embeddings per token: {embeddings}")

    # update the data dict
    data_dict['NT_TF'] = embeddings

    # garbage collect
    del embeddings, torch_outs, tokens_ids, attention_mask, rna_seq, codon_seq
    gc.collect()

    # save the new file
    # with open(train_files[i], 'wb') as f:
    #     pkl.dump(data_dict, f)
    #     f.close()

    # break

# test files
for i in range(len(test_files)):
    print(i)
    with open(test_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    codon_seq = np.asarray(data_dict['sequence'])

    rna_seq = getRNASeq(codon_seq)
    # print(rna_seq)
    tokens_ids = tokenizer.batch_encode_plus(rna_seq, return_tensors="pt")["input_ids"]

    # Compute the embeddings
    attention_mask = tokens_ids != tokenizer.pad_token_id
    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
    )

    # Compute sequences embeddings
    embeddings = torch_outs['hidden_states'][-1].detach().numpy()
    # remove the CLS tokens
    embeddings = embeddings[:, 1:, :]
    # take mean of the embeddings for each codon
    embeddings = embeddings.mean(axis=1)
    print(f"Embeddings shape: {embeddings.shape}")
    # print(f"Embeddings per token: {embeddings}")

    # update the data dict
    data_dict['NT_TF'] = embeddings

    # save the new file
    # with open(test_files[i], 'wb') as f:
    #     pkl.dump(data_dict, f)
    #     f.close()

    # break

# val files
for i in range(len(val_files)):
    print(i)
    with open(val_files[i], 'rb') as f:
        data_dict = pkl.load(f)
        f.close()
    
    codon_seq = np.asarray(data_dict['sequence'])

    rna_seq = getRNASeq(codon_seq)
    print(len(rna_seq))
    tokens_ids = tokenizer.batch_encode_plus(rna_seq, return_tensors="pt")["input_ids"]

    # Compute the embeddings
    attention_mask = tokens_ids != tokenizer.pad_token_id
    torch_outs = model(
        tokens_ids,
        attention_mask=attention_mask,
        encoder_attention_mask=attention_mask,
        output_hidden_states=True
    )

    # Compute sequences embeddings
    embeddings = torch_outs['hidden_states'][-1].detach().numpy()
    # remove the CLS tokens
    embeddings = embeddings[:, 1:, :]
    # take mean of the embeddings for each codon
    embeddings = embeddings.mean(axis=1)
    print(f"Embeddings shape: {embeddings.shape}")
    # print(f"Embeddings per token: {embeddings}")

    # update the data dict
    data_dict['NT_TF'] = embeddings

    # save the new file
    # with open(val_files[i], 'wb') as f:
    #     pkl.dump(data_dict, f)
    #     f.close()

    # break