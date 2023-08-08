# libraries
import numpy as np
import time
import torch 
import gc
from torch import nn, Tensor 
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from scipy.stats import pearsonr, spearmanr
from os import listdir
from os.path import isfile, join
import math
import pickle as pkl
from torch.utils.data import Dataset, DataLoader

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


class TransformerModel(nn.Module):
    def __init__(self, num_feats: int, nhead: int, d_hid: int,
                 nlayers: int, mult_factor, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_hid, dropout)
        encoder_layers = TransformerEncoderLayer(d_hid, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(num_feats, d_hid)
        self.d_hid = d_hid
        self.decoder = nn.Linear(d_hid, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # print(src.shape)
        src = self.encoder(src) * math.sqrt(self.d_hid)
        src = torch.unsqueeze(src, 1)
        src = self.pos_encoder(src)
        src = torch.squeeze(src, 1)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)

        return output

class TransformerBiLSTMModel(nn.Module):
    def __init__(self, num_feats: int, nhead: int, d_hid: int,
                 nlayers: int, mult_factor, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_hid, dropout)
        encoder_layers = TransformerEncoderLayer(d_hid, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(num_feats, d_hid)
        self.d_hid = d_hid
        
        self.lstm = nn.LSTM(d_hid, d_hid, 4, bidirectional = True, dropout = dropout, batch_first=True)

        self.decoder = nn.Linear(d_hid * 2, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # print(src.shape)
        src = self.encoder(src) * math.sqrt(self.d_hid)
        src = torch.unsqueeze(src, 1)
        src = self.pos_encoder(src)
        src = torch.squeeze(src, 1)
        output = self.transformer_encoder(src, src_mask)
        output, (h_n, c_n) = self.lstm(output)
        output = self.decoder(output)

        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    start_time = time.time()
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        inputs, labels, lengths, condition = data
        loss_batch = 0.
        for x in range(len(lengths)):
            inputs_sample = inputs[x][:lengths[x], :].float().to(device)
            labels_sample = labels[x][:lengths[x]].float().to(device)
            src_mask = generate_square_subsequent_mask(inputs_sample.shape[1]).float().to(device)
            inputs_sample = torch.squeeze(inputs_sample, 0)

            labels_sample = labels_sample.float().to(device)
            labels_sample = torch.squeeze(labels_sample, 0)
            outputs = model(inputs_sample, src_mask)
            outputs = torch.squeeze(outputs, 0)
            outputs = torch.squeeze(outputs, 1)
        
            loss_batch += criterion(outputs, labels_sample)

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels_sample.cpu().detach().numpy()

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            corr_s, _ = spearmanr(y_true_det, y_pred_det)
            
            pearson_corr_lis.append(corr_p)
            spearman_corr_lis.append(corr_s)

            del inputs_sample, labels_sample, outputs, y_pred_det, y_true_det
            torch.cuda.empty_cache()
            gc.collect()

        loss_batch /= len(lengths)
        
        del inputs, labels, lengths, condition
        torch.cuda.empty_cache()
        gc.collect()

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        total_loss += loss_batch.item() * loss_mult_factor

        if (i) % (10) == 0:
            logger.info(f'| samples trained: {(i+1)*bs:5d} | train (intermediate) loss: {total_loss/(i+1):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')
        
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
            inputs = inputs.to(device)
            inputs = torch.squeeze(inputs, 0)

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
