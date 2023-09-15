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
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class RBDataset(Dataset):
    def __init__(self, list_of_file_paths):
        self.list_of_file_paths = list_of_file_paths
        self.max_len = 10000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)
        
        X_pad = np.zeros((self.max_len, 804))
        X = np.asarray(data_dict['X'])
        # normalize X (not a good idea)
        # X[:,15:15+768] = (X[:,15:15+768] - X[:,15:15+768].min(0)) / X[:,15:15+768].ptp(0)
        # print(X)
        X_pad[:X.shape[0], :] = X

        y_pad = np.empty(self.max_len)
        y_pad.fill(-1)
        # normalize y (not a good idea)
        y = np.absolute(data_dict['y']) 
        # y = (y - y.min(0)) / y.ptp(0)
        y = y * 1e+6
        # print(y)
        # print(y.shape)
        y_pad[:y.shape[0]] = y
        
        return X_pad, y_pad, len(X)

class RBDataset_NoBS(Dataset):
    def __init__(self, list_of_file_paths):
        self.list_of_file_paths = list_of_file_paths
        self.max_len = 10000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)

        nt = np.asarray(data_dict['nt'])
        cbert = np.asarray(data_dict['cbert'])
        conds = np.asarray(data_dict['conds'])
        depr_vec = np.asarray(data_dict['depr_vec'])
        X_seq = np.asarray(data_dict['sequence'])
        if np.max(X_seq) > 64:
            Nnt_present = True 
        else:
            Nnt_present = False
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
        X_ft = np.concatenate((cbert, X_ft), axis=1)
        X_ft = np.concatenate((nt, X_ft), axis=1)
        gene = data_dict['gene']
        # print(X_pad.shape, y_pad.shape)
        # print(X.shape, y.shape)

        y = [float(val) for val in data_dict['y']]
        y = np.asarray(y)
        y = np.absolute(y)
        # y = np.log1p(y)
        
        return X_ft, y, self.list_of_file_paths[index].split('/')[-1].split('_')[1], Nnt_present, self.list_of_file_paths[index].split('/')[-1]

class RBDataset_NoBS_withGene(Dataset):
    def __init__(self, list_of_file_paths):
        self.list_of_file_paths = list_of_file_paths
        self.max_len = 10000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)
        # print(data_dict.keys())
        
        nt = np.asarray(data_dict['nt'])
        cbert = np.asarray(data_dict['cbert'])
        conds = np.asarray(data_dict['conds'])
        depr_vec = np.asarray(data_dict['depr_vec'])
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
        X_ft = np.concatenate((cbert, X_ft), axis=1)
        X_ft = np.concatenate((nt, X_ft), axis=1)
        gene = data_dict['gene']
        # print(X_pad.shape, y_pad.shape)
        # print(X.shape, y.shape)

        y = [float(val) for val in data_dict['y']]
        y = np.asarray(y)
        y = np.absolute(y)
        
        return X_ft, y, self.list_of_file_paths[index].split('/')[-1].split('_')[0], gene, self.list_of_file_paths[index].split('/')[-1]

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs):
        super().__init__()

        # self.embedding = nn.Embedding(84, 256)

        # lstm model
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout, batch_first=True)

        # params
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.bs = bs
        self.n_layers = n_layers

        # functions
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # linear layers
        self.lin1 = nn.Sequential(
            nn.Linear((hidden_dim * 2 if bidirectional else hidden_dim), 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lin2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lin3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.out_lin = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, src):

        # sequence analysis using LSTM
        h_0 = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)
        outputs, (final_hidden, final_cell) = self.lstm(src, (h_0, c_0))

        # append conds_vec to outputs
        # lin 1
        # x = torch.cat((outputs, conds_vec), dim=2)
        x = self.lin1(outputs)
        # x, _ = self.attn1(x, x, x)

        # lin 2
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.lin2(x)
        # x, _ = self.attn2(x, x, x)

        # lin 3
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.lin3(x)

        # out lin
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.out_lin(x)

        return x

class BiLSTMModel_Explain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs):
        super().__init__()

        # self.embedding = nn.Embedding(84, 256)

        # lstm model
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout, batch_first=True)

        # params
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.bs = bs
        self.n_layers = n_layers

        # functions
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # linear layers
        self.lin1 = nn.Sequential(
            nn.Linear((hidden_dim * 2 if bidirectional else hidden_dim), 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lin2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lin3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.out_lin = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, src):

        # sequence analysis using LSTM
        h_0 = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)
        outputs, (final_hidden, final_cell) = self.lstm(src, (h_0, c_0))

        # append conds_vec to outputs
        # lin 1
        # x = torch.cat((outputs, conds_vec), dim=2)
        x = self.lin1(outputs)
        # x, _ = self.attn1(x, x, x)

        # lin 2
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.lin2(x)
        # x, _ = self.attn2(x, x, x)

        # lin 3
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.lin3(x)

        # out lin
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.out_lin(x)

        return x, final_hidden, final_cell

def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        optimizer.zero_grad()
        # inputs, labels, len_seq = data
        # len_seq = len_seq.numpy()
        inputs, labels = data
        if len(labels) == bs:
            # print(len_seq)
            # labels to numpy and check the thresold 
            inputs = inputs.float().to(device)
            # inputs = torch.unsqueeze(inputs, 2)
            # print(inputs.shape, labels.shape)

            labels = labels.float().to(device)
            labels = torch.squeeze(labels, 0)
            
            outputs = model(inputs)
            outputs = torch.squeeze(outputs, 0)
            # outputs = torch.squeeze(outputs, 2)
            outputs = torch.squeeze(outputs, 1)
            # print(outputs.shape, labels.shape)
            
            # a = torch.zeros((1,5)).to(device)
            # loss = criterion(a,a)

            # for j in range(bs):
            #     loss += criterion(outputs[j][:len_seq[j]], labels[j][:len_seq[j]]) 

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * loss_mult_factor

            # for j in range(bs):
            #     y_pred_det = outputs[j][:len_seq[j]].cpu().detach().numpy()
            #     y_true_det = labels[j][:len_seq[j]].cpu().detach().numpy()

            #     corr_p, _ = pearsonr(y_true_det, y_pred_det)
            #     corr_s, _ = spearmanr(y_true_det, y_pred_det)
                
            #     pearson_corr_lis.append(corr_p)
            #     spearman_corr_lis.append(corr_s)

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels.cpu().detach().numpy()

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            corr_s, _ = spearmanr(y_true_det, y_pred_det)
            
            pearson_corr_lis.append(corr_p)
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
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            # inputs, labels, len_seq = data
            # len_seq = len_seq.numpy()
            inputs, labels, condition, nont, transcript = data
            if len(labels) == bs:
                inputs = inputs.float().to(device)
                lab_np = labels.cpu().detach().numpy()
                non_zero = np.count_nonzero(lab_np)
                perc = non_zero / len(lab_np)
                # inputs = torch.unsqueeze(inputs, 2)
                # print(inputs.shape, labels.shape)

                labels = labels.float().to(device)
                labels = torch.squeeze(labels, 0)
                
                outputs = model(inputs)
                outputs = torch.squeeze(outputs, 0)
                # outputs = torch.squeeze(outputs, 2)
                outputs = torch.squeeze(outputs, 1)
                
                # a = torch.zeros((1,5)).to(device)
                # loss = criterion(a,a)

                # for j in range(bs):
                #     loss += criterion(outputs[j][:len_seq[j]], labels[j][:len_seq[j]]) 

                loss = criterion(outputs, labels)

                total_loss += loss.item()

                # for j in range(bs):
                #     y_pred_det = outputs[j][:len_seq[j]].cpu().detach().numpy()
                #     y_true_det = labels[j][:len_seq[j]].cpu().detach().numpy()

                #     corr_p, _ = pearsonr(y_true_det, y_pred_det)
                    
                #     corr_lis.append(corr_p)

                y_pred_det = outputs.cpu().detach().numpy()
                y_true_det = labels.cpu().detach().numpy()

                corr_p, _ = pearsonr(y_true_det, y_pred_det)
                
                corr_lis.append(corr_p)

    logger.info(f'| PR: {np.mean(corr_lis):5.5f} |')

    return total_loss / (len(val_dataloader) - 1)

def hidden_states_eval(model: nn.Module, test_dataloader, device, logger) -> float:
    print("Explaining Hidden States")
    # model.eval()
    # conds_list = []
    # fin_hid_list = []
    # fin_cell_list = []
    # transcript_list = []
    # with torch.no_grad():
    #     for i, data in enumerate(test_dataloader):
    #         inputs, labels, cond_tag, nont, transcript = data
    #         inputs = inputs.float().to(device)

    #         labels = labels.float().to(device)
    #         labels = torch.squeeze(labels, 0)
            
    #         outputs, fin_hid, fin_cell = model(inputs)
    #         # print(outputs.shape, fin_hid.shape, fin_cell.shape)
    #         fin_hid = torch.squeeze(fin_hid, 1)
    #         # flatten the hidden state
    #         fin_hid = torch.flatten(fin_hid)
    #         fin_cell = torch.squeeze(fin_cell, 1)
    #         # flatten the cell state
    #         fin_cell = torch.flatten(fin_cell)

    #         # add np arrays to lists 
    #         # print(cond_tag)
    #         cond_tag = cond_tag[0]
    #         print(cond_tag, fin_hid.shape, fin_cell.shape)
    #         if cond_tag == 'CTRL':
    #             conds_list.append(0)
    #         elif cond_tag == 'LEU':
    #             conds_list.append(1)
    #         elif cond_tag == 'ILE':
    #             conds_list.append(2)
    #         elif cond_tag == 'VAL':
    #             conds_list.append(3)
    #         elif cond_tag == 'LEU-ILE':
    #             conds_list.append(4)
    #         elif cond_tag == 'LEU-ILE-VAL':
    #             conds_list.append(5)
    #         fin_hid_list.append(fin_hid.cpu().detach().numpy())
    #         fin_cell_list.append(fin_cell.cpu().detach().numpy())

    #         transcript_list.append(transcript[0])

    # # save the hidden states and cell states and conds
    # np.save('hidden_states.npy', fin_hid_list)
    # np.save('cell_states.npy', fin_cell_list)
    # np.save('conds.npy', conds_list)
    # np.save('transcripts.npy', transcript_list)

    # load npy files
    file_path = ''
    fin_hid_list = np.load(file_path + 'hidden_states.npy')
    conds_list = np.load(file_path + 'conds.npy')
    fin_cell_list = np.load(file_path + 'cell_states.npy')
    
    # UMAP the hidden states
    reducer = umap.UMAP(n_epochs=1, random_state=42)
    hid_states = np.asarray(fin_hid_list)
    hid_states = reducer.fit_transform(hid_states)

    # make a hid state plot
    plt.figure(figsize=(10, 8))
    # sns scatter plot
    data = pd.DataFrame(hid_states, columns=['x', 'y'])
    data['conds'] = conds_list
    custom_colors = ['#27ae60', '#e67e22', '#2980b9', '#e74c3c', '#8e44ad', '#2c3e50']
    sns.scatterplot(
        x = 'x',
        y = 'y',
        hue = 'conds',
        data = data,
        palette = custom_colors,
        legend = 'full',
        alpha = 1
        )

    # scatter = plt.scatter(
    #     x = hid_states[:,0],
    #     y = hid_states[:,1],
    #     c = conds_list
    #     )
    classes = ['CTRL', 'LEU', 'ILE', 'VAL', 'LEU-ILE', 'LEU-ILE-VAL']
    plt.legend(labels=classes, fontsize=15, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    # plt.title('Hidden States UMAP', fontsize=24)
    # tight layout
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/hidden_states.png')
    plt.clf()

    # UMAP the cell states
    reducer = umap.UMAP()
    cell_states = np.asarray(fin_cell_list)
    cell_states = reducer.fit_transform(cell_states)

    # make a cell state plot
    data = pd.DataFrame(hid_states, columns=['x', 'y'])
    data['conds'] = conds_list
    scatter = sns.scatterplot(
        x = 'x',
        y = 'y',
        hue = 'conds',
        data = data,
        palette = custom_colors,
        legend = 'full',
        alpha = 1
        )
    # plt.figure(figsize=(8, 8))
    # scatter = plt.scatter(
    #     x = cell_states[:,0],
    #     y = cell_states[:,1],
    #     c = conds_list
    #     )
    classes = ['CTRL', 'LEU', 'ILE', 'VAL', 'LEU-ILE', 'LEU-ILE-VAL']
    plt.legend(labels=classes, fontsize=15, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    # plt.title('Cell States UMAP', fontsize=24)
    # tight layout
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/cell_states.png')
    plt.clf()






