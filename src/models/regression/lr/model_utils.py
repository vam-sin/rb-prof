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
        self.max_len = 10000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)

        X = np.asarray(data_dict['X'])

        # split up
        nt = X[:,:15]
        cbert = X[:,15:15+768]
        # t5 = X[:,15+768:15+768+1024]
        # conds = X[:,15+768+1024:15+768+1024+21]

        # add one hot for ctrl as well (that was all 0's maybe that was the issue)
        conds = X[:,15+768:15+768+20]
        conds_new = np.zeros((X.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        conds_new[:,0:20] = conds
        if 'CTRL' in self.list_of_file_paths[index]:
            # set last one to 1
            conds_new[:,20] = 1

        depr_codons = X[:,15+768+20:15+768+20+1]
        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_codons), axis=1)

        # concatenate
        X_ft = np.concatenate((nt, conds_fin), axis=1)

        # y
        y = np.absolute(data_dict['y'])
        y = y * 1e+6

        # print(self.list_of_file_paths[index], conds_fin[0])

        condition = self.list_of_file_paths[index].split('/')[-1].split('_')[1]
        
        return X_ft, y, condition

class LRModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, src):
        x = self.linear(src)
        x = F.relu(x)

        return x

def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        optimizer.zero_grad()
        inputs, labels, condition = data
        inputs = inputs.float().to(device)
        inputs = torch.squeeze(inputs, 0)
        # inputs = inputs.permute(1, 0)

        labels = labels.float().to(device)
        labels = torch.squeeze(labels, 0)
        
        outputs = model(inputs)
        outputs = torch.squeeze(outputs, 1)

        loss = criterion(outputs, labels) 

        loss.backward()
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
            inputs, labels, condition = data
            inputs = inputs.float().to(device)
            inputs = torch.squeeze(inputs, 0)
            # inputs = inputs.permute(1, 0)

            labels = labels.float().to(device)
            labels = torch.squeeze(labels, 0)
            
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, 1)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels.cpu().detach().numpy()

            corr_p, _ = pearsonr(y_true_det, y_pred_det)

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
