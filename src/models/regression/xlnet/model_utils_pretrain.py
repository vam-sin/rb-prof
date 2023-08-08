# libraries
import os
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import csv
from transformers import Trainer
from torchmetrics import PearsonCorrCoef
import itertools
import pickle as pkl

codon_table_ds06 = {
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

inv_codon_table_ds06 = {v: k for k, v in codon_table_ds06.items()}

# inverse of the dict
number_to_codon = {idx+1:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G', 'N'], repeat=3))}
codon_to_number = {v:k for k,v in number_to_codon.items()} # to convert sequence to numbers

class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcc_l = PearsonCorrCoef().to(torch.device('cuda'))
        
    def forward(self, pred, actual):
        # print(pred.shape, actual.shape)
        pred = torch.squeeze(pred, dim=0)
        actual = torch.squeeze(actual, dim=0)
        return -1 * self.pcc_l(pred, actual)

class RiboDatasetLiver(Dataset):
    def __init__(self, ribo_data_dirpath: str):
        with open(os.path.join(ribo_data_dirpath, "counts.csv"), "r") as read_obj:
            self.counts = list(csv.reader(read_obj))
        with open(os.path.join(ribo_data_dirpath, "sequences.csv"), "r") as read_obj:
            self.sequences = list(csv.reader(read_obj))

        # perc_annot list
        self.perc_annot = []
        for i in range(len(self.counts)):
            num_nonzero = np.count_nonzero(np.array(self.counts[i]).astype(np.float))
            self.perc_annot.append(num_nonzero / len(self.counts[i]))
        
        thresh = 0.6
        # remove sequences with less than thresh% annotation from the counts and sequences files
        self.counts = [self.counts[i] for i in range(len(self.counts)) if self.perc_annot[i] >= thresh]
        self.sequences = [self.sequences[i] for i in range(len(self.sequences)) if self.perc_annot[i] >= thresh]

        # pad all sequences with pad token 125
        # max_len = max([len(seq) for seq in self.sequences])
        # for i in range(len(self.sequences)):
        #     self.sequences[i] = self.sequences[i] + (max_len - len(self.sequences[i])) * ['125']

        # # pad all counts with 0.0
        # for i in range(len(self.counts)):
        #     self.counts[i] = self.counts[i] + (max_len - len(self.counts[i])) * ['0.0']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(list(map(int, self.sequences[idx]))), torch.tensor(
            list(map(float, self.counts[idx]))
        )

class RBDataset_NoBS(Dataset):
    def __init__(self, list_of_file_paths, dataset_name, feature_list):
        self.list_of_file_paths = list_of_file_paths
        self.dataset_name = dataset_name
        self.feature_list = feature_list
        self.max_len = 10000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)

        # get data
        X_seq = np.asarray(data_dict['sequence'])

        # convert numbers to codons
        X_seq = [inv_codon_table_ds06[el] for el in X_seq]

        # convert codons to numbers
        X_seq = [codon_to_number[el] for el in X_seq]

        y = [float(val) for val in data_dict['y']]
        y = np.asarray(y)
        y = np.absolute(y)

        X_seq = torch.tensor(X_seq, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float32)

        return X_seq, y

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.squeeze(logits, dim=2)
        # logits = torch.squeeze(logits, dim=0)
        # labels = torch.squeeze(labels, dim=0)
        # print(logits.shape, labels.shape)
        # print("logits", logits)
        # print("labels", labels)
        loss_fnc = PCCLoss()
        loss = loss_fnc(logits, labels)
        return (loss, outputs) if return_outputs else loss 