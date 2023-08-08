# libraries
import numpy as np
from sklearn.model_selection import train_test_split
import torch 
from torch import nn
from scipy.stats import pearsonr, spearmanr 
from os import listdir
from os.path import isfile, join
import math
import pickle as pkl 
from torch.autograd import Variable
from torch.utils.data import Dataset

conds_dict = {'CTRL': 0, 'LEU': 1, 'ILE': 2, 'VAL': 3, 'LEU-ILE': 4, 'LEU-ILE-VAL': 5}

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

        # get data
        X_seq = np.asarray(data_dict['sequence'])

        # # features
        nt = np.asarray(data_dict['nt'])
        cbert = np.asarray(data_dict['cbert'])

        conds_new = np.zeros((nt.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        # if 'CTRL' in self.list_of_file_paths[index]:
        # set last one to 1 (all are CTRL in this case)
        conds_new[:,20] = 1
        # depr vec is all zeros
        depr_vec = np.zeros((nt.shape[0], 1))

        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_vec), axis=1)

        # combine features
        X_ft = conds_fin
        if 'cbert' in self.feature_list:
            X_ft = np.concatenate((cbert, X_ft), axis=1)
        if 'nt' in self.feature_list:
            X_ft = np.concatenate((nt, X_ft), axis=1)

        # y
        # y = np.absolute(data_dict['y'])[:len_]
        y = np.absolute(data_dict['y'])
        # print(y)
        # normalize with means and stds
        # y = (y - np.mean(y)) / np.std(y)
        # y = y
        # min max norm
        # y = (y - np.min(y)) / (np.max(y) - np.min(y))
        # replace nan with 0
        # y = np.nan_to_num(y)
        # # replace inf with 0
        # y = np.where(y == np.inf, 0, y)

        condition = 'CTRL'
        # print(self.list_of_file_paths[index])
        # print('gene', data_dict['gene'])
        
        return X_ft, conds_fin, y, condition

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs):
        super().__init__()

        # # embedding layer 
        # self.embedding = nn.Embedding(84, 256)

        # # context embedding
        # self.context_embedding = nn.Embedding(21, 783)

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
    
    def forward(self, src, condition):

        # # remove depr vector which is the last feature
        # depr_vec = src[:, :, -1]
        # depr_vec = torch.squeeze(depr_vec, axis=0)
        # src = src[:, :, :-1]
        # # print(src.shape)
        # # print(depr_vec.shape)

        # # context embedding
        # # get one the conds_fin
        # # print(condition)
        # # print(condition.shape)
        # conds_embed = self.context_embedding(condition)
        # # print(conds_embed.shape)

        # # iterate over depr_vec and if it is 1, add the conds embed to the src of that codon
        # for i in range(len(depr_vec)):
        #     if depr_vec[i] == 1:
        #         src[0][i] += conds_embed

        # embedding
        # src = self.embedding(src)
        # src = torch.cat((src, conds_fin), dim=2)

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

def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        optimizer.zero_grad()
        inputs, conds_fin, labels, condition = data
        # perc_sequence annotation
        len_ = inputs.shape[1]
        num_non_zero_labels = torch.sum(labels != 0.0)
        perc_non_zero_labels = num_non_zero_labels / (len_)
        if perc_non_zero_labels > 0.0:
            inputs = inputs.float().to(device)
            conds_fin = conds_fin.float().to(device)
            # condition_vec = torch.tensor(conds_dict[condition[0]]).long().to(device)

            labels = labels.float().to(device)
            labels = torch.squeeze(labels, 0)
            
            outputs = model(inputs, conds_fin)
            outputs = torch.squeeze(outputs, 0)
            outputs = torch.squeeze(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * loss_mult_factor

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels.cpu().detach().numpy()

            # # check if nans in inputs
            # nan_torch_inp = torch.isnan(inputs)
            # nan_torch_inp = torch.sum(nan_torch_inp)
            # if nan_torch_inp > 0:
            #     print("nan in inputs")
            #     print(inputs)

            # # check if nans in outputs
            # outputs = torch.isnan(outputs)
            # outputs = torch.sum(outputs)
            # if outputs > 0:
            #     print("nan in outputs")
            #     print(outputs)

            # print(inputs, conds_fin)

            # print(y_pred_det, y_true_det)
            # print(y_pred_det)

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            corr_s, _ = spearmanr(y_true_det, y_pred_det)
            # print(corr_p)
            
            pearson_corr_lis.append(corr_p)
            spearman_corr_lis.append(corr_s)

        if (i) % (100) == 0:
            logger.info(f'| samples trained: {(i+1)*bs:5d} | train (intermediate) loss: {total_loss/((i+1)*bs):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')
    
    logger.info(f'Epoch Train Loss: {total_loss/len(train_dataloder): 5.10f} | train pr: {np.mean(pearson_corr_lis):5.10f} | train sr: {np.mean(spearman_corr_lis):5.10f} |')

    return total_loss/len(train_dataloder), np.mean(pearson_corr_lis), np.mean(spearman_corr_lis)

def evaluate(model: nn.Module, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs) -> float:
    print("Evaluating")
    model.eval()
    total_loss = 0. 
    corr_lis = []
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs, conds_fin, labels, condition = data
            len_ = inputs.shape[1]
            num_non_zero_labels = torch.sum(labels != 0.0)
            perc_non_zero_labels = num_non_zero_labels / (len_)
            if perc_non_zero_labels > 0.0:
                inputs = inputs.float().to(device)
                # embeds
                # inputs = inputs.long().to(device)
                conds_fin = conds_fin.float().to(device)
                # condition_vec = torch.tensor(conds_dict[condition[0]]).long().to(device)

                labels = labels.float().to(device)
                labels = torch.squeeze(labels, 0)
                
                outputs = model(inputs, conds_fin)
                outputs = torch.squeeze(outputs, 0)
                outputs = torch.squeeze(outputs, 1)

                loss = criterion(outputs, labels)

                total_loss += loss.item()

                y_pred_det = outputs.cpu().detach().numpy()
                y_true_det = labels.cpu().detach().numpy()

                corr_p, _ = pearsonr(y_true_det, y_pred_det)

                condition = condition[0]
                
                corr_lis.append(corr_p)

                # # append preds and filename to dict 
                # file_preds[filename[0]] = y_pred_det

    logger.info(f'| Full PR: {np.mean(corr_lis):5.5f} |')
    
    return total_loss / (len(val_dataloader) - 1), np.mean(corr_lis)
