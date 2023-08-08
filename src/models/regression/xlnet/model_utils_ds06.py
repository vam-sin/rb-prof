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
import pickle as pkl

class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcc_l = PearsonCorrCoef().to(torch.device('cuda'))
        
    def forward(self, pred, actual):
        # print(pred.shape, actual.shape)
        pred = torch.squeeze(pred, dim=0)
        actual = torch.squeeze(actual, dim=0)
        return -1 * self.pcc_l(pred, actual)

class RBDataset_NoBS(Dataset):
    def __init__(self, list_of_file_paths, dataset_name, feature_list):
        self.list_of_file_paths = list_of_file_paths
        self.dataset_name = dataset_name
        self.feature_list = feature_list
        self.max_len = 6000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)

        # # features
        if self.dataset_name == 'DS06':
            nt = np.asarray(data_dict['nt'])
            cbert = np.asarray(data_dict['cbert'])
            t5 = np.asarray(data_dict['t5'])
            lem = np.asarray(data_dict['lem'])
            mlm_cdna_nt = np.asarray(data_dict['mlm_cdna_nt_pbert'])
            mlm_cdna_nt_idai = np.asarray(data_dict['mlm_cdna_nt_idai'])
            af2 = np.asarray(data_dict['AF2-SS'])
            conds = np.asarray(data_dict['conds'])
            depr_vec = np.asarray(data_dict['depr_vec'])
            geom = np.asarray(data_dict['geom'])
            codon_epa_encodings = np.asarray(data_dict['codon_epa_encodings'])
            X_seq = np.asarray(data_dict['sequence'])

            if 'AF2-SS' in self.feature_list or 't5' in self.feature_list or 'geom' in self.feature_list:
                len_ = af2.shape[0]
                nt = nt[:len_, :]
                cbert = cbert[:len_, :]
                lem = lem[:len_, :]
                mlm_cdna_nt = mlm_cdna_nt[:len_, :]
                mlm_cdna_nt_idai = mlm_cdna_nt_idai[:len_, :]
                conds = np.asarray(data_dict['conds'])[:len_, :]
                depr_vec = np.asarray(data_dict['depr_vec'])[:len_, :]

            if 'codon_epa_encodings' in self.feature_list:
                nt = nt[2:, :]
                conds = np.asarray(data_dict['conds'])[2:, :]
                depr_vec = np.asarray(data_dict['depr_vec'])[2:, :]

        elif self.dataset_name == 'DS06_Liver06':
            nt = np.asarray(data_dict['nt'])
            cbert = np.asarray(data_dict['cbert'])
            conds = np.asarray(data_dict['conds'])
            depr_vec = np.asarray(data_dict['depr_vec'])

        elif self.dataset_name == 'DS04':
            X = np.asarray(data_dict['X'])
            nt = X[:,0:15]
            cbert = X[:,15:15+768]
            conds = X[:,15+768:15+768+20]
            depr_vec = X[:,15+768+20:15+768+20+1]
            X_seq = np.asarray(data_dict['sequence'])

        conds_new = np.zeros((nt.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        conds_new[:,0:20] = conds
        if 'CTRL' in self.list_of_file_paths[index]:
            # set last one to 1
            conds_new[:,20] = 1

        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_vec, depr_vec, depr_vec, depr_vec), axis=1)

        # combine features
        X_ft = conds_fin
        if 'AF2-SS' in self.feature_list:
            X_ft = np.concatenate((af2, X_ft), axis=1)
        if 'mlm_cdna_nt_idai' in self.feature_list:
            X_ft = np.concatenate((mlm_cdna_nt_idai, X_ft), axis=1)
        if 'mlm_cdna_nt_pbert' in self.feature_list:
            X_ft = np.concatenate((mlm_cdna_nt, X_ft), axis=1)
        if 'lem' in self.feature_list:
            X_ft = np.concatenate((lem, X_ft), axis=1)
        if 't5' in self.feature_list:
            X_ft = np.concatenate((t5, X_ft), axis=1)
        if 'cbert' in self.feature_list:
            X_ft = np.concatenate((cbert, X_ft), axis=1)
        if 'nt' in self.feature_list:
            X_ft = np.concatenate((nt, X_ft), axis=1)
        if 'geom' in self.feature_list:
            X_ft = np.concatenate((geom, X_ft), axis=1)
        if 'codon_epa_encodings' in self.feature_list:
            X_ft = np.concatenate((codon_epa_encodings, X_ft), axis=1)

        # y
        # y = np.absolute(data_dict['y'])[:len_]
        # print(data_dict['y'])
        y = [float(val) for val in data_dict['y']]
        y = np.asarray(y)
        y = np.absolute(y)
        # y = y
        # min max norm
        # y = (y - np.min(y)) / (np.max(y) - np.min(y))

        if 'codon_epa_encodings' in self.feature_list:
            y = y[2:]

        if 'AF2-SS' in self.feature_list or 't5' in self.feature_list or 'geom' in self.feature_list:
            y = y[:len_]

        condition = self.list_of_file_paths[index].split('/')[-1].split('_')[1]

        X_ft = torch.tensor(X_ft, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # generate attention mask
        attention_mask = torch.zeros((self.max_len, self.max_len))
        attention_mask[:X_ft.shape[0], :X_ft.shape[0]] = 1

        # pad the features to max length
        X_ft = torch.nn.functional.pad(X_ft, (0,0,0,self.max_len - X_ft.shape[0]), 'constant', 0)

        # pad labels with -100
        y = torch.nn.functional.pad(y, (0,self.max_len - y.shape[0]), 'constant', -100)

        if 'cembeds' in self.feature_list:
            return X_seq, conds_fin, y, condition

        # print(X_ft.shape, y.shape, attention_mask.shape)
        # print(X_ft, y, attention_mask)
        
        return [X_ft, y, attention_mask]

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # inputs["inputs_embeds"] = torch.squeeze(inputs["inputs_embeds"], dim=0)
        # print(inputs["inputs_embeds"].shape)
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.squeeze(logits, dim=2)
        loss_fnc = PCCLoss()
        loss = loss_fnc(logits, labels)
        return (loss, outputs) if return_outputs else loss 