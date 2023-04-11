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
from torch.utils.data import Dataset, DataLoader

class RBDataset(Dataset):
    def __init__(self, list_of_file_paths):
        self.list_of_file_paths = list_of_file_paths

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)
        
        X = np.asarray(data_dict['X'])
        y = np.absolute(data_dict['y']) * 1e+6
        
        return X, y

def process_sample(input_file, mult_factor):
    '''
    conducts the processing of a single sample
    1. loads the file
    2. splits the file into X, and y
    3. generates mask
    4. returns all these
    '''
    with open(input_file, 'rb') as f:
        data_dict = pkl.load(f)
        
    X = np.asarray(data_dict['X'])
    y = np.absolute(data_dict['y']) * 1e+6

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

def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        optimizer.zero_grad()
        inputs, labels = data
        # print(inputs.shape)
        src_mask = generate_square_subsequent_mask(inputs.shape[1]).float().to(device)
        inputs = inputs.float().to(device)
        inputs = torch.squeeze(inputs, 0)
        # inputs = torch.unsqueeze(inputs, 2)
        # print(inputs.shape, labels.shape)

        labels = labels.float().to(device)
        labels = torch.squeeze(labels, 0)
        
        outputs = model(inputs, src_mask)
        outputs = torch.squeeze(outputs, 0)
        outputs = torch.squeeze(outputs, 1)
        
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * loss_mult_factor

        y_pred_det = outputs.cpu().detach().numpy()
        y_true_det = labels.cpu().detach().numpy()

        corr_p, _ = pearsonr(y_true_det, y_pred_det)
        corr_s, _ = spearmanr(y_true_det, y_pred_det)
        
        pearson_corr_lis.append(corr_p)
        spearman_corr_lis.append(corr_s)

        if (i) % (bs*50) == 0:
            logger.info(f'| samples trained: {i+1:5d} | train (intermediate) loss: {total_loss/(i+1):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')
        
    logger.info(f'Epoch Train Loss: {total_loss/len(train_dataloder): 5.10f} | train pr: {np.mean(pearson_corr_lis):5.10f} | train sr: {np.mean(spearman_corr_lis):5.10f} |')

def evaluate(model: nn.Module, val_dataloader, device, mult_factor, criterion, logger) -> float:
    print("Evaluating")
    model.eval()
    total_loss = 0. 
    corr_lis = []
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs, labels = data
            inputs = inputs.float().to(device)
            src_mask = generate_square_subsequent_mask(inputs.shape[1]).float().to(device)
            inputs = inputs.float().to(device)
            inputs = torch.squeeze(inputs, 0)
            # print(inputs.shape, labels.shape)

            labels = labels.float().to(device)
            labels = torch.squeeze(labels, 0)
            
            outputs = model(inputs, src_mask)
            outputs = torch.squeeze(outputs, 0)
            outputs = torch.squeeze(outputs, 1)
            
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels.cpu().detach().numpy()

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            
            corr_lis.append(corr_p)

    logger.info(f'| PR: {np.mean(corr_lis):5.5f} |')

    return total_loss / (len(val_dataloader) - 1)
