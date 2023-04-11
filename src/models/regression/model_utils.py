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
    X = arr['feature_vec_DNABERT'][:,:15] # NT
    # X = arr['feature_vec_DNABERT'][:,15:783] # C-BERT
    # X = arr['feature_vec'][:,:115] # NT + C
    # X = arr['feature_vec'][:,115:1139] # T5
    # X = arr['feature_vec'][:,:1139] # NT, C, T5
    # X = arr['feature_vec'][:,1139:] # LRS
    # X = arr['feature_vec_DNABERT'] # full
    y = np.absolute(arr['counts'])

    # count vectors
    # counts per million
    y = y * 1e+6

    return X, y

class TransformerModel(nn.Module):
    def __init__(self, num_feats: int, d_model: int, nhead: int, d_hid: int,
                nlayers: int,  mult_factor: float, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        '''Transformer Encoder Layer: Attn + FFN 
        d_model: num_feats from input
        nhead: num of multihead attention models
        d_hid: dimension of the FFN
        how many nts distance is the attention heads looking at.
        '''
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # Transformer Encoder Model: made up on multiple encoder layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(num_feats, d_model)
        self.mult_factor = mult_factor
        self.d_model = d_model 
        self.decoder1 = nn.Linear(d_model, 128)
        self.fc = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1 
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.relu(self.transformer_encoder(src, src_mask))
        output = self.decoder1(output)
        output = self.dropout(self.relu(output))
        output = self.fc(output)
        # output = self.sigmoid(output) * self.mult_factor
 
        return output 

def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 17000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def train(model: nn.Module, tr_train, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    log_interval = 100
    start_time = time.time() 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i in range(0, len(tr_train), bs):
        a = torch.zeros((1,5)).to(device)
        loss = criterion(a,a)
        for b in range(i, min(i+bs, len(tr_train))):
            X, y = process_sample(tr_train[b], mult_factor)
            if len(X) < 10000: # not going to fit in gpu unless
                seq_len = len(X)
                x_in = torch.from_numpy(X).float().to(device)
                src_mask = generate_square_subsequent_mask(seq_len).float().to(device)
                y_pred = torch.flatten(model(x_in, src_mask))
                y_true = torch.flatten(torch.from_numpy(y)).float().to(device)

                y_pred_imp_seq = []
                y_true_imp_seq = []
                for x in range(len(y_pred.cpu().detach().numpy())):
                    y_pred_imp_seq.append(y_pred[x].item())
                    y_true_imp_seq.append(y_true[x].item())

                corr_p, _ = pearsonr(y_true_imp_seq, y_pred_imp_seq)
                corr_s, _ = spearmanr(y_true_imp_seq, y_pred_imp_seq)
                
                pearson_corr_lis.append(corr_p)
                spearman_corr_lis.append(corr_s)
                # print(corr_p, corr_s)
                
                loss += criterion(y_pred, y_true) * loss_mult_factor # multiplying to make the loss bigger

                # remove from GPU device
                del x_in
                del src_mask
                del y_true
                del y_pred
                gc.collect()
                torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        # print(i+1)
        if (i) % (bs*50) == 0:
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
                src_mask = generate_square_subsequent_mask(seq_len).to(device)
                x_in = torch.from_numpy(X).float().to(device)
                y_pred = torch.flatten(model(x_in, src_mask))
                y_true = torch.flatten(torch.from_numpy(y)).to(device)

                y_pred_imp_seq = []
                y_true_imp_seq = []
                for x in range(len(y_pred.cpu().detach().numpy())):
                    y_pred_imp_seq.append(y_pred[x].item())
                    y_true_imp_seq.append(y_true[x].item())

                corr, _ = pearsonr(y_true_imp_seq, y_pred_imp_seq)
                corr_lis.append(corr)

                loss = criterion(y_pred, y_true)

                total_loss += loss.item()

                del x_in
                del src_mask
                del y_pred
                del y_true
                del loss
                
                gc.collect()
                torch.cuda.empty_cache()

    logger.info(f'| PR: {np.mean(corr_lis):5.5f} |')

    return total_loss / (len(tr_val) - 1)
