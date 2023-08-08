# libraries
import numpy as np
from sklearn.model_selection import train_test_split
import random
import gc
import math 
import copy
import time
import torch 
from torch import nn, Tensor 
import torch.nn.functional as F 
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr 
from os import listdir
from os.path import isfile, join
import math
import sys
import logging
import pickle as pkl 
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import MultiheadAttention

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

        # # get data
        # X_seq = np.asarray(data_dict['sequence'])

        # # features
        if self.dataset_name == 'DS06':
            nt = np.asarray(data_dict['nt'])
            cbert = np.asarray(data_dict['cbert'])
            # t5 = np.asarray(data_dict['t5'])
            # lem = np.asarray(data_dict['lem'])
            # mlm_cdna_nt = np.asarray(data_dict['mlm_cdna_nt_pbert'])
            # mlm_cdna_nt_idai = np.asarray(data_dict['mlm_cdna_nt_idai'])
            # af2 = np.asarray(data_dict['AF2-SS'])
            conds = np.asarray(data_dict['conds'])
            depr_vec = np.asarray(data_dict['depr_vec'])
            # geom = np.asarray(data_dict['geom'])
            # codon_epa_encodings = np.asarray(data_dict['codon_epa_encodings'])

            # if 'AF2-SS' in self.feature_list or 't5' in self.feature_list or 'geom' in self.feature_list:
            #     len_ = geom.shape[0]
            #     nt = nt[:len_, :]
            #     # cbert = cbert[:len_, :]
            #     # lem = lem[:len_, :]
            #     # mlm_cdna_nt = mlm_cdna_nt[:len_, :]
            #     # mlm_cdna_nt_idai = mlm_cdna_nt_idai[:len_, :]
            #     conds = np.asarray(data_dict['conds'])[:len_, :]
            #     depr_vec = np.asarray(data_dict['depr_vec'])[:len_, :]

            # if 'codon_epa_encodings' in self.feature_list:
            #     nt = nt[2:, :]
            #     conds = np.asarray(data_dict['conds'])[2:, :]
            #     depr_vec = np.asarray(data_dict['depr_vec'])[2:, :]

        # elif self.dataset_name == 'DS04':
        #     X = np.asarray(data_dict['X'])
        #     nt = X[:,0:15]
        #     cbert = X[:,15:15+768]
        #     conds = X[:,15+768:15+768+20]
        #     depr_vec = X[:,15+768+20:15+768+20+1]

        conds_new = np.zeros((nt.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        conds_new[:,0:20] = conds
        if 'CTRL' in self.list_of_file_paths[index]:
            # set last one to 1
            conds_new[:,20] = 1

        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_vec), axis=1)

        # combine features
        X_ft = conds_fin
        # if 'AF2-SS' in self.feature_list:
        #     X_ft = np.concatenate((af2, X_ft), axis=1)
        # if 'mlm_cdna_nt_idai' in self.feature_list:
        #     X_ft = np.concatenate((mlm_cdna_nt_idai, X_ft), axis=1)
        # if 'mlm_cdna_nt_pbert' in self.feature_list:
        #     X_ft = np.concatenate((mlm_cdna_nt, X_ft), axis=1)
        # if 'lem' in self.feature_list:
        #     X_ft = np.concatenate((lem, X_ft), axis=1)
        # if 't5' in self.feature_list:
        #     X_ft = np.concatenate((t5, X_ft), axis=1)
        if 'cbert' in self.feature_list:
            X_ft = np.concatenate((cbert, X_ft), axis=1)
        if 'nt' in self.feature_list:
            X_ft = np.concatenate((nt, X_ft), axis=1)
        # if 'geom' in self.feature_list:
        #     X_ft = np.concatenate((geom, X_ft), axis=1)
        # if 'codon_epa_encodings' in self.feature_list:
        #     X_ft = np.concatenate((codon_epa_encodings, X_ft), axis=1)

        # y
        # y = np.absolute(data_dict['y'])[:len_]
        # y = np.absolute(data_dict['y'])
        # y = y
        # min max norm
        # y = (y - np.min(y)) / (np.max(y) - np.min(y))
        y = [float(val) for val in data_dict['y']]
        y = np.asarray(y)
        y = np.absolute(y)

        # if 'codon_epa_encodings' in self.feature_list:
        #     y = y[2:]

        # if 'AF2-SS' in self.feature_list or 't5' in self.feature_list or 'geom' in self.feature_list:
        #     y = y[:len_]

        condition = self.list_of_file_paths[index].split('/')[-1].split('_')[1]

        # if 'cembeds' in self.feature_list:
        #     return X_seq, conds_fin, y, condition

        # X_ft = torch.tensor(X_ft, dtype=torch.float32)
        # y = torch.tensor(y, dtype=torch.float32)
        
        return X_ft, conds_fin, y, condition

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        # self.attn1 = MultiheadAttention(embed_dim = input_dim, num_heads = 4, dropout = dropout)

        # self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.cnn1 = nn.Conv1d(in_channels = input_dim, out_channels = 512, kernel_size = 7, stride = 1, padding = 3)
        self.cnn2 = nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 7, stride = 1, padding = 3)
        self.cnn1_2 = nn.Conv1d(in_channels = input_dim, out_channels = 256, kernel_size = 7, stride = 1, padding = 3)
        self.cnn3 = nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size = 7, stride = 1, padding = 3)
        self.cnn4 = nn.Conv1d(in_channels = 256, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.cnn3_4 = nn.Conv1d(in_channels = 256, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.fnn = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, output_dim),
        nn.ReLU()
        )
    
    def forward(self, src):
        # print(src.shape)
        src = torch.squeeze(src, 0)
        src = src.permute(1, 0)
        x = self.cnn1(src)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.dropout(x)
        x1 = self.cnn1_2(src)
        # print(x.shape, x1.shape)
        x = x + x1
        x_mid = self.relu(x)
        x = self.cnn3(x_mid)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn4(x)
        x = self.dropout(x)
        x2 = self.cnn3_4(x_mid)
        x = x + x2
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(1, 0)
        x = self.fnn(x)

        return x

def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        optimizer.zero_grad()
        inputs, conds_fin, labels, condition = data
        inputs = inputs.float().to(device)
        # inputs = torch.squeeze(inputs, 0)
        # inputs = inputs.permute(0, 2, 1)

        labels = labels.float().to(device)
        labels = torch.squeeze(labels, 0)
        
        outputs = model(inputs)
        outputs = torch.squeeze(outputs, 1)

        loss = criterion(outputs, labels)

        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item() * loss_mult_factor

        y_pred_det = outputs.cpu().detach().numpy()
        y_true_det = labels.cpu().detach().numpy()

        corr_p, _ = pearsonr(y_true_det, y_pred_det)
        corr_s, _ = spearmanr(y_true_det, y_pred_det)
        
        pearson_corr_lis.append(corr_p)
        # print(corr_p)
        spearman_corr_lis.append(corr_s)


        # print(i)

        if (i) % (100) == 0:
            logger.info(f'| samples trained: {(i+1)*bs:5d} | train (intermediate) loss: {total_loss/((i+1)*bs):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')
    
    logger.info(f'Epoch Train Loss: {total_loss/len(train_dataloder): 5.10f} | train pr: {np.mean(pearson_corr_lis):5.10f} | train sr: {np.mean(spearman_corr_lis):5.10f} |')

    return total_loss/len(train_dataloder), np.mean(pearson_corr_lis), np.mean(spearman_corr_lis)

def evaluate(model: nn.Module, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs) -> float:
    print("Evaluating")
    model.eval()
    total_loss = 0. 
    corr_lis = []
    ctrl_corr_lis = []
    leu_corr_lis = []
    ile_corr_lis = []
    val_corr_lis = []
    leu_ile_corr_lis = []
    leu_ile_val_corr_lis = []
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs, conds_fin, labels, condition = data
            inputs = inputs.float().to(device)

            labels = labels.float().to(device)
            labels = torch.squeeze(labels, 0)
            
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, 1)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels.cpu().detach().numpy()

            corr_p, _ = pearsonr(y_true_det, y_pred_det)

            condition = condition[0]

            if condition == 'CTRL':
                ctrl_corr_lis.append(corr_p)
            elif condition == 'LEU':
                leu_corr_lis.append(corr_p)
            elif condition == 'ILE':
                ile_corr_lis.append(corr_p)
            elif condition == 'VAL':
                val_corr_lis.append(corr_p)
            elif condition == 'LEU-ILE':
                leu_ile_corr_lis.append(corr_p)
            elif condition == 'LEU-ILE-VAL':
                leu_ile_val_corr_lis.append(corr_p)
            
            corr_lis.append(corr_p)

    logger.info(f'| Full PR: {np.mean(corr_lis):5.5f} |')
    logger.info(f'| CTRL PR: {np.mean(ctrl_corr_lis):5.5f} |')
    logger.info(f'| LEU PR: {np.mean(leu_corr_lis):5.5f} |')
    logger.info(f'| ILE PR: {np.mean(ile_corr_lis):5.5f} |')
    logger.info(f'| VAL PR: {np.mean(val_corr_lis):5.5f} |')
    logger.info(f'| LEU_ILE PR: {np.mean(leu_ile_corr_lis):5.5f} |')
    logger.info(f'| LEU_ILE_VAL PR: {np.mean(leu_ile_val_corr_lis):5.5f} |')

    # save preds
    # with open('file_preds.pkl', 'wb') as f:
    #     pkl.dump(file_preds, f)
    
    return total_loss / (len(val_dataloader) - 1), np.mean(corr_lis)
