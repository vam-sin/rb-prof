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
from torch.nn.parameter import Parameter

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

        # # features
        if self.dataset_name == 'DS06':
            nt = np.asarray(data_dict['nt'])
            cbert = np.asarray(data_dict['cbert'])
            t5 = np.asarray(data_dict['t5'])
            lem = np.asarray(data_dict['lem'])
            mlm_cdna_nt = np.asarray(data_dict['mlm_cdna_nt_pbert'])
            mlm_cdna_nt_idai = np.asarray(data_dict['mlm_cdna_nt_idai'])
            af2 = np.asarray(data_dict['AF2-SS'])
            conds = np.asarray(data_dict['conds'])
            depr_vec = np.asarray(data_dict['depr_vec'])

            if 't5' in self.feature_list:
                len_ = af2.shape[0]
                nt = nt[:len_, :]
                cbert = cbert[:len_, :]
                lem = lem[:len_, :]
                mlm_cdna_nt = mlm_cdna_nt[:len_, :]
                mlm_cdna_nt_idai = mlm_cdna_nt_idai[:len_, :]
                conds = np.asarray(data_dict['conds'])[:len_, :]
                depr_vec = np.asarray(data_dict['depr_vec'])[:len_, :]

        elif self.dataset_name == 'DS04':
            X = np.asarray(data_dict['X'])
            nt = X[:,0:15]
            cbert = X[:,15:15+768]
            conds = X[:,15+768:15+768+20]
            depr_vec = X[:,15+768+20:15+768+20+1]


        conds_new = np.zeros((nt.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        conds_new[:,0:20] = conds
        if 'CTRL' in self.list_of_file_paths[index]:
            # set last one to 1
            conds_new[:,20] = 1

        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_vec), axis=1)

        # combine features
        X_ft = depr_vec
        if 'AF2-SS' in self.feature_list:
            X_ft = np.concatenate((af2, X_ft), axis=1)
        if 'mlm_cdna_nt_idai' in self.feature_list:
            X_ft = np.concatenate((mlm_cdna_nt_idai, X_ft), axis=1)
        if 'mlm_cdna_nt_pbert' in self.feature_list:
            X_ft = np.concatenate((mlm_cdna_nt, X_ft), axis=1)
        if 'lem' in self.feature_list:
            X_ft = np.concatenate((lem, X_ft), axis=1)
        if 't5' in self.feature_list:
            X_ft = np.concatenate((t5, X_ft), axis=1)
        if 'cbert' in self.feature_list:
            X_ft = np.concatenate((cbert, X_ft), axis=1)
        if 'nt' in self.feature_list:
            X_ft = np.concatenate((nt, X_ft), axis=1)

        # y
        # y = np.absolute(data_dict['y'])[:len_]
        y = np.absolute(data_dict['y'])
        # y = y

        condition = self.list_of_file_paths[index].split('/')[-1].split('_')[1]

        if condition == 'CTRL':
            conds_one_hot = 0
        elif condition == 'LEU':
            conds_one_hot = 1
        elif condition == 'VAL':
            conds_one_hot = 2
        elif condition == 'ILE':
            conds_one_hot = 3
        elif condition == 'LEU-ILE':
            conds_one_hot = 4
        elif condition == 'LEU-ILE-VAL':
            conds_one_hot = 5
        
        return X_ft, conds_one_hot, y, condition

# add more context to the the nn RNN Cell
class FactorCell(nn.Module):
    def __init__(self, num_units, embedding_size, context_embedding_size, low_rank_adaptation = False, rank = 10, dropout = 0.1):
        super().__init__()

        self.num_units = num_units
        self._forget_bias = 1.0
        self.embedding_size = embedding_size
        self.context_embedding_size = context_embedding_size
        self.low_rank_adaptation = low_rank_adaptation
        self.rank = rank
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        self.input_size = num_units + embedding_size

        self.W = torch.nn.Parameter(torch.Tensor(self.input_size, 3 * self.num_units))
        self.bias = torch.nn.Parameter(torch.Tensor(3 * self.num_units))

        if low_rank_adaptation:
            # left adapt
            self.left_adapt_generator = torch.nn.Parameter(torch.Tensor(self.context_embedding_size, self.input_size * self.rank))

            # right adapt
            self.right_adapt_generator = torch.nn.Parameter(torch.Tensor(self.context_embedding_size, self.rank * 3 * self.num_units))

    def forward(self, inputs, context_embedding, state):
        c, h = state 
        # print('c', c.shape)
        # print('h', h.shape)
        # print('inputs', inputs.shape)
        
        h = torch.squeeze(h, 0)
        h = torch.squeeze(h, 0)
        h = h.repeat(1, inputs.shape[1], 1)
        # print('h reshaped', h.shape)
        the_input = torch.cat([inputs, h], 2)
        # print('the_input', the_input.shape)
        # print('W', self.W.shape)
        result = torch.matmul(the_input, self.W)
        # print('result', result.shape)

        if self.low_rank_adaptation:
            # matmul context embedding with left_adapt_generator to get left_adapt
            # print('context_embedding', context_embedding.shape)
            # print('left_adapt_generator', self.left_adapt_generator.shape)
            left_adapt_unshaped = torch.matmul(context_embedding, self.left_adapt_generator)
            left_adapt = torch.reshape(left_adapt_unshaped, (-1, self.input_size, self.rank))
            # left_adapt = left_adapt.permute(0, 2, 1)

            right_adapt_unshaped = torch.matmul(context_embedding, self.right_adapt_generator)
            right_adapt = torch.reshape(right_adapt_unshaped, (-1, self.rank, 3 * self.num_units))
            # right_adapt = right_adapt.permute(0, 2, 1)

            # print('left_adapt', left_adapt.shape)
            # print('right_adapt', right_adapt.shape)
            
            inter = torch.matmul(the_input, left_adapt)
            # print('result inter', result.shape)
            final = torch.matmul(inter, right_adapt)

        result += final

        # print('result fin', result.shape)
        # print('bias', self.bias.shape)
        result = result + self.bias

        # split result into 3 parts j, f, o
        j, f, o = torch.split(result, split_size_or_sections=[256, 256, 256], dim=2)

        # apply activation functions
        j = self.relu(j)
        j = self.dropout_layer(j)
        
        # sigmoid    
        forget_gate = torch.sigmoid(f + self._forget_bias)
        input_gate = 1.0 - forget_gate
        new_c = (c * forget_gate + input_gate * j)
        new_h = self.relu(new_c) * torch.sigmoid(o)

        new_state = (new_c, new_h)

        return new_h, new_state

class LRRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs):
        super().__init__()

        # condition embedding
        self.cond_emb = nn.Embedding(21, hidden_dim)

        # lstm model
        self.lrr = FactorCell(hidden_dim, input_dim, hidden_dim, low_rank_adaptation=True, rank=10, dropout=dropout)

        # params
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.bs = bs
        self.n_layers = n_layers

        # functions
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # lstm out linear layers
        self.lin1 = nn.Sequential(
            nn.Linear((hidden_dim), 256),
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

        # out linear layer
        self.out_lin = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, src, conds_one_hot):
        # condition embedding
        conds_emb = self.cond_emb(conds_one_hot)
        # ---------------------------------------------
        # LSTM Network

        # sequence analysis using LSTM
        h_0 = Variable(torch.zeros(1, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(1, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)

        # low rank rnn 
        output, (final_hidden_state, final_cell_state) = self.lrr(src, conds_emb, (h_0, c_0)) # (bs, seq_len, hidden)

        # ---------------------------------------------

        # lin 1
        # output = output + conds_emb 
        x = self.lin1(output)
        # conds_emb = self.lin_conds1(conds_emb)

        # lin 2
        # x = x + conds_emb
        x = self.lin2(x)
        # conds_emb = self.lin_conds2(conds_emb)

        # lin 3
        # x = x + conds_emb
        x = self.lin3(x)
        # conds_emb = self.lin_conds3(conds_emb)

        # out lin
        # x = x + conds_emb
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
        inputs, conds_one_hot, labels, condition = data

        inputs = inputs.float().to(device)
        conds_one_hot = conds_one_hot.long().to(device)

        labels = labels.float().to(device)
        labels = torch.squeeze(labels, 0)
        
        outputs = model(inputs, conds_one_hot)
        outputs = torch.squeeze(outputs, 0)
        outputs = torch.squeeze(outputs, 1)
        # print(outputs)

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
            inputs, conds_fin, labels, condition = data
            inputs = inputs.float().to(device)
            conds_fin = conds_fin.long().to(device)

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
