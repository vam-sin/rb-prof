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
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer 
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
    filename_ = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon_DNABERT/' + input_key + '.npz'
    # filename_ = '/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/seq_annot_final/final_dataset_codon/' + input_key
    arr = np.load(filename_, allow_pickle=True)['arr_0'].item()
    X = arr['feature_vec_DNABERT'][:,1807:] # LRS
    # X = arr['feature_vec_DNABERT'] # full
    y = np.absolute(arr['counts'])

    # count vectors
    # counts per million
    y = y * mult_factor

    return X, y

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
        # print(self.pe.shape, self.pe[:x.size(0)].shape)
        # print(x, self.pe[:x.size(0)])
        x = self.dropout(x + self.pe[:x.size(0)])
        return x

class TransformerModel(nn.Module):
    def __init__(self, intoken, outtoken, nhead, hidden, enc_layers, dec_layers, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(intoken, hidden) 
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Linear(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.hidden = hidden

        self.transformer = nn.Transformer(d_model = hidden, nhead = nhead, num_encoder_layers = enc_layers, num_decoder_layers = dec_layers, dim_feedforward = hidden, dropout=dropout, activation="relu")
        self.fc_out = nn.Linear(hidden, outtoken)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.src_mask = None 
        self.trg_mask = None 
        self.memory_mask = None 

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1 
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        # print(mask)
        return mask 
    
    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        src = self.relu(self.encoder(src)) * math.sqrt(self.hidden)
        src = self.pos_encoder(src)
        # print(src.shape)

        trg = self.relu(self.decoder(trg)) * math.sqrt(self.hidden)
        trg = self.pos_decoder(trg)
        # print(trg.shape)

        output = self.transformer(src, trg, tgt_mask = self.trg_mask)
        # print(output.shape)
        output = self.fc_out(output)
        # print(output.shape)
        output = self.sigmoid(output)
        # print(output.shape)

        return output

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
        optimizer.zero_grad()

        for b in range(i, min(i+bs, len(tr_train))):
            X, y = process_sample(tr_train[b], mult_factor)
            if len(X) < 10000: # not going to fit in gpu unless
                seq_len = len(X)
                x_in = torch.from_numpy(X).float().to(device)
                # src_mask = generate_square_subsequent_mask(seq_len).float().to(device)
                y_true = torch.from_numpy(np.expand_dims(y, axis=1)).float().to(device)
                
                y_pred = torch.squeeze(model(x_in, y_true), dim=1)
                
                y_true = torch.squeeze(y_true, dim=1)
                # print(y_pred.shape, y_true.shape)

                y_pred_imp_seq = []
                y_true_imp_seq = []
                for x in range(len(y_pred.cpu().detach().numpy())):
                    y_pred_imp_seq.append(y_pred[x].item())
                    y_true_imp_seq.append(y_true[x].item())

                corr_p, _ = pearsonr(y_true_imp_seq, y_pred_imp_seq)
                corr_s, _ = spearmanr(y_true_imp_seq, y_pred_imp_seq)
                # print(y_pred_imp_seq)
                
                pearson_corr_lis.append(corr_p)
                spearman_corr_lis.append(corr_s)
                # print(corr_p, corr_s)
                # print(y_pred.shape, y_true.shape)
                
                loss += criterion(y_pred, y_true) * loss_mult_factor # multiplying to make the loss bigger

                # remove from GPU device
                del x_in
                # del src_mask
                del y_true
                del y_pred
                gc.collect()
                torch.cuda.empty_cache()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        # print(i+1)
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
                # src_mask = generate_square_subsequent_mask(seq_len).float().to(device)
                y_true = torch.from_numpy(np.expand_dims(y, axis=1)).float().to(device)
                
                y_pred = torch.squeeze(model(x_in, y_true), dim=1)
                
                y_true = torch.squeeze(y_true, dim=1)

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
                # del src_mask
                del y_pred
                del y_true
                del loss
                
                gc.collect()
                torch.cuda.empty_cache()

    logger.info(f'| PR: {np.mean(corr_lis):5.5f} |')

    return total_loss / (len(tr_val) - 1)
