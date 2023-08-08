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
    def __init__(self, list_of_file_paths):
        self.list_of_file_paths = list_of_file_paths
        self.max_len = 6000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)

        X = np.asarray(data_dict['X'])
        # print(X.shape)
        X_seq = data_dict['sequence']

        # # split up
        nt = X[:,:15]
        cbert = X[:,15:15+768]
        # t5 = X[:,15+768:15+768+1024]
        # conds = X[:,15+768+1024:15+768+1024+21]

        # MLM Embeds
        # X = np.asarray(data_dict['X_MLM_cDNA_NT'])
        # # print(X.shape)

        # # split up
        # nt_mlm_embeds = X[:,:384]
        # conds = X[:,384:384+20]


        # add one hot for ctrl as well (that was all 0's maybe that was the issue)
        conds = X[:,15+768:15+768+20]
        conds_new = np.zeros((X.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        conds_new[:,0:20] = conds
        if 'CTRL' in self.list_of_file_paths[index]:
            # set last one to 1
            conds_new[:,20] = 1

        depr_codons = X[:,384+20:384+20+1]
        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_codons), axis=1)

        # concatenate
        X_ft = np.concatenate((nt, cbert, conds_fin), axis=1)

        # y
        y = np.absolute(data_dict['y'])
        y = y * 1e+6

        # print(self.list_of_file_paths[index], conds_fin[0])

        condition = self.list_of_file_paths[index].split('/')[-1].split('_')[1]
        conds_vec = conds_new[0]
        # print(conds_vec)
        # print(self.list_of_file_paths[index], condition)
        # print(X_ft.shape, conds_vec.shape, y.shape, condition)

        if condition == 'CTRL':
            condition_value = 0
        elif condition == 'LEU':
            condition_value = 1
        elif condition == 'ILE':
            condition_value = 2
        elif condition == 'VAL':
            condition_value = 3
        elif condition == 'LEU-ILE':
            condition_value = 4
        elif condition == 'LEU-ILE-VAL':
            condition_value = 5

        # print(X_ft.shape, y.shape, condition_value)

        # pad X_ft and y
        X_pad = np.zeros((self.max_len, X_ft.shape[1]))
        y_pad = np.zeros((self.max_len))
        X_pad[:X_ft.shape[0], :] = X_ft
        y_pad[:y.shape[0]] = y

        return X_pad, y_pad, len(X_ft), condition

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()

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
        src = torch.squeeze(src, 0)
        src = src.permute(1, 0)
        x = self.cnn1(src)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.dropout(x)
        x1 = self.cnn1_2(src)
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
        inputs, labels, lengths, conditions_lis = data
        # print(inputs.shape, labels.shape, lengths.shape)
        loss_batch = 0.
        for x in range(len(lengths)):
            inputs_sample = inputs[x][:lengths[x], :].float().to(device)
            inputs_sample = torch.unsqueeze(inputs_sample, 0)
            labels_sample = labels[x][:lengths[x]].float().to(device)

            labels_sample = torch.squeeze(labels_sample, 0)
            # print(inputs_sample.shape, labels_sample.shape)

            # condition_value_sample = conditions_lis[x].long().to(device)
            
            outputs = model(inputs_sample)
            
            outputs = torch.squeeze(outputs, 1)

            loss_batch += criterion(outputs, labels_sample)

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels_sample.cpu().detach().numpy()

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            corr_s, _ = spearmanr(y_true_det, y_pred_det)
            
            pearson_corr_lis.append(corr_p)
            spearman_corr_lis.append(corr_s)
        
        # print(loss_batch.item())

        # criterion when you give the whole batch to it does the division by batch size
        loss_batch /= len(lengths)
        
        loss_batch.backward()
        optimizer.step()
        total_loss += loss_batch.item() * loss_mult_factor

        if (i) % (10) == 0:
            logger.info(f'| samples trained: {(i+1)*bs:5d} | train (intermediate) loss: {total_loss/((i+1)):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')
    
    logger.info(f'Epoch Train Loss: {total_loss/len(train_dataloder): 5.10f} | train pr: {np.mean(pearson_corr_lis):5.10f} | train sr: {np.mean(spearman_corr_lis):5.10f} |')

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
            inputs, labels, length, condition = data

            inputs = inputs[:, :length[0], :].float().to(device)

            labels = labels[:, :length[0]].float().to(device)
            labels = torch.squeeze(labels, 0)

            # condition_sample = condition[0].long().to(device)
            
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, 0)
            outputs = torch.squeeze(outputs, 1)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels.cpu().detach().numpy()

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            # print(corr_p)

            # print(condition)
            # print(y_pred_det)
            # print(y_true_det)

            # print(condition, condition[0])
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

    return total_loss / (len(val_dataloader) - 1)
