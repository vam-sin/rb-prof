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

class RBDataset(Dataset):
    def __init__(self, list_of_file_paths):
        self.list_of_file_paths = list_of_file_paths

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        filename_ = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon_DNABERT/' + self.list_of_file_paths[index] + '.npz.npz'
        # print(self.list_of_file_paths[index])
        arr = np.load(filename_, allow_pickle=True)['arr_0'].item()
        X = arr['feature_vec_DNABERT'][:,:15]
        # print(X.shape)
        
        # print(codon_seq)
        y = np.absolute(arr['counts']) * 1e+6

        # print(codon_seq.shape, y.shape)
        
        return X, y

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        # self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout if n_layers > 1 else 0, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128)
        self.fc = nn.Linear(128, output_dim)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, src):
        # src = self.embedding(src)
        h_0 = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, 1, self.hidden_dim).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, 1, self.hidden_dim).cuda()) # (1, bs, hidden)
        outputs, (final_hidden, final_cell) = self.lstm(src, (h_0, c_0))
        x = self.relu(outputs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc(x)
        # x = self.sigmoid(x)

        return x

def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.float().to(device)
        # inputs = torch.unsqueeze(inputs, 2)
        # print(inputs.shape, labels.shape)

        labels = labels.float().to(device)
        labels = torch.squeeze(labels, 0)
        
        outputs = model(inputs)
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

def evaluate(model: nn.Module, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger) -> float:
    print("Evaluating")
    model.eval()
    total_loss = 0. 
    corr_lis = []
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs, labels = data
            inputs = inputs.float().to(device)
            # inputs = torch.unsqueeze(inputs, 2)
            # print(inputs.shape, labels.shape)

            labels = labels.float().to(device)
            labels = torch.squeeze(labels, 0)
            
            outputs = model(inputs)
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
