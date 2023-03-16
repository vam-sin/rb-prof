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

def process_sample(input_key, mult_factor):
    '''
    conducts the processing of a single sample
    1. loads the file
    2. splits the file into X, and y
    3. generates mask
    4. returns all these
    '''
    filename_ = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon_DNABERT/' + input_key + '.npz.npz'
    # filename_ = '/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon/' + input_key
    arr = np.load(filename_, allow_pickle=True)['arr_0'].item()
    # X = arr['feature_vec'][:,:15] # NT
    # X = arr['feature_vec_DNABERT'][:,15:783] # C-BERT
    # X = arr['feature_vec'][:,:115] # NT + C
    # X = arr['feature_vec'][:,115:1139] # T5
    # X = arr['feature_vec'][:,:1139] # NT, C, T5
    # X = arr['feature_vec'][:,1139:] # LRS
    # X = arr['feature_vec_DNABERT'] # full
    X = arr['feature_vec_DNABERT'][:,:15]
    y = np.absolute(arr['counts'])

    # count vectors
    # counts per million
    y = y * mult_factor

    return X, y

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout if n_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src):
        h_0 = Variable(torch.zeros(1, 1, self.hidden_dim).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(1, 1, self.hidden_dim).cuda()) # (1, bs, hidden)
        outputs, (final_hidden, final_cell) = self.lstm(src, (h_0, c_0))
        # print(outputs.shape, final_hidden.shape, final_cell.shape)
        x = self.fc(outputs)
        x = self.sigmoid(x)
        # print(x.shape)

        return x

def train(model: nn.Module, tr_train, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    log_interval = 100
    start_time = time.time() 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i in range(0, len(tr_train), bs):
        model.zero_grad(set_to_none=True)
        a = torch.zeros((1,5)).to(device)
        loss = criterion(a,a)
        for b in range(i, min(i+bs, len(tr_train))):
            X, y = process_sample(tr_train[b], mult_factor)
            if len(X) < 10000: # not going to fit in gpu unless
                seq_len = len(X)
                x_in = torch.from_numpy(X).float().to(device)
                x_in = torch.unsqueeze(x_in, 0)

                y_pred = model(x_in)
                y_pred = torch.squeeze(y_pred, 0)
                y_pred = torch.squeeze(y_pred, 1)
                y_true = torch.flatten(torch.from_numpy(y)).float().to(device)

                y_pred_det = y_pred.cpu().detach().numpy()

                corr_p, _ = pearsonr(y, y_pred_det)
                corr_s, _ = spearmanr(y, y_pred_det)
                
                pearson_corr_lis.append(corr_p)
                spearman_corr_lis.append(corr_s)
                
                loss += criterion(y_pred, y_true) * loss_mult_factor # multiplying to make the loss bigger

                # remove from GPU device
                del x_in
                # del src_mask
                del y_true
                del y_pred
                gc.collect()
                torch.cuda.empty_cache()
                
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i) % (bs) == 0:
            logger.info(f'| samples trained: {i+1:5d} | train (intermediate) loss: {total_loss/(i+1):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')

    logger.info(f'Epoch Train Loss: {total_loss/len(tr_train): 5.10f} | train pr: {np.mean(pearson_corr_lis):5.10f} | train sr: {np.mean(spearman_corr_lis):5.10f} |')

def evaluate(model: nn.Module, tr_val, device, mult_factor, criterion, logger) -> float:
    print("Evaluating")
    model.eval()
    total_loss = 0 
    corr_lis = []
    with torch.no_grad():
        for i in range(len(tr_val)):
            X, y = process_sample(tr_val[i], mult_factor)
            if len(X) < 10000:
                seq_len = len(X)
                x_in = torch.from_numpy(X).float().to(device)
                x_in = torch.unsqueeze(x_in, 0)

                y_pred = model(x_in)
                y_pred = torch.squeeze(y_pred, 0)
                y_pred = torch.squeeze(y_pred, 1)
                y_true = torch.flatten(torch.from_numpy(y)).float().to(device)

                y_pred_det = y_pred.cpu().detach().numpy()

                corr_p, _ = pearsonr(y, y_pred_det)
                
                corr_lis.append(corr_p)

                loss = criterion(y_pred, y_true) * 1e+6
                # print(loss.item(), corr)

                total_loss += loss.item()

                del x_in
                # del src_mask
                del y_pred
                del y_true
                del loss
                
                gc.collect()
                torch.cuda.empty_cache()

    logger.info(f'| PR: {np.mean(corr_lis):5.5f} |')

    return total_loss / (len(tr_val) - 1)
