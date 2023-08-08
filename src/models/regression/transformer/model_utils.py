# libraries
import numpy as np
import random
import gc
import math 
import torch 
from torch import nn, Tensor 
from scipy.stats import pearsonr, spearmanr 
from os import listdir
from os.path import isfile, join
import math
import pickle as pkl 
from torch.utils.data import Dataset

# reproducability
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# # global variables for the special tokens
sos_token_inp = 0
eos_token_inp = 86
sos_token_out = 1e-6
eos_token_out = 1e-6

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
        X_seq_orig = np.asarray(data_dict['sequence'])
        # add special tokens
        X_seq_final = [sos_token_inp] + list(X_seq_orig) + [eos_token_inp]
        X_seq_final = np.asarray(X_seq_final)

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
        X_ft = conds_fin
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

        # add one hot for ctrl as well (that was all 0's maybe that was the issue)
        conds_new = np.zeros((nt.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        conds_new[:,0:20] = conds
        if 'CTRL' in self.list_of_file_paths[index]:
            # set last one to 1
            conds_new[:,20] = 1

        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_vec), axis=1)
        sos_eos_conds = conds_fin[0]
        sos_eos_conds = np.expand_dims(sos_eos_conds, axis=0)
        # add eos and sos conds
        conds_fin = np.concatenate((sos_eos_conds, conds_fin, sos_eos_conds), axis=0)


        # X_ft = np.concatenate((nt, cbert, conds_fin), axis=1)

        # y
        # y = np.absolute(data_dict['y'])[:len_]
        y = np.absolute(data_dict['y'])
        y = y + 1e-6
        # add special tokens
        y = [sos_token_out] + list(y) + [eos_token_out]
        y = np.asarray(y)
        y = y * 1e+6

        condition = self.list_of_file_paths[index].split('/')[-1].split('_')[1]
        
        return X_seq_final, conds_fin, y, condition

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
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)

        return self.dropout(x)

def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dims, n_heads, num_enc_layers, num_dec_layers, dim_ff, dropout):
        super().__init__()

        self.model_dims = model_dims

        self.encoder_embeds = nn.Embedding(87, model_dims-22)
        self.decoder_embeds = nn.Linear(1, model_dims)
        self.posEmbed = PositionalEncoding(model_dims, dropout)
        self.transformer_model = nn.Transformer(model_dims, nhead=n_heads, num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers, dim_feedforward=dim_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU() # switched to ReLU 

        self.linear_layers = nn.Sequential(
            nn.Linear(model_dims, 1),
            nn.ReLU()
        )
    
    def forward(self, src, tgt, conds_additional, src_mask=None, tgt_mask=None):
        src = self.encoder_embeds(src)
        # append conds 
        src = torch.cat((src, conds_additional), dim=2) 
        # src = src * math.sqrt(self.model_dims)
        # src = self.posEmbed(src)
        src = src.permute(1, 0, 2)
        # tgt = self.decoder_embeds(tgt) * math.sqrt(self.model_dims)
        # tgt = self.posEmbed(tgt)
        tgt = self.decoder_embeds(tgt)
        tgt = tgt.permute(1, 0, 2)

        if src_mask == None:
            src_mask = generate_square_subsequent_mask(src.shape[0]).to(src.device)
        if tgt_mask == None:
            tgt_mask = generate_square_subsequent_mask(tgt.shape[0]).to(tgt.device)

        output = self.transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)
        output = self.linear_layers(output)

        return output

def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        optimizer.zero_grad()
        inputs, conds, labels, condition = data
        if inputs.shape[1] <= 5000:
            inputs = inputs.long().to(device)

            # process the labels
            labels = labels.float().to(device)
            labels = torch.unsqueeze(labels, 2)

            # shift the labels and send to the decoder
            dec_input = labels[:, :-1]

            # conds
            conds = conds.float().to(device)

            # get predictions
            outputs = model(src = inputs, tgt = dec_input, conds_additional = conds)

            # remove last output token because it is the eos token
            outputs = outputs[:, :-1, :]
            outputs = torch.squeeze(outputs, 0)
            outputs = torch.squeeze(outputs, 1)

            # remove special tokens from labels
            labels = labels[:, 1:-1]
            labels = torch.squeeze(labels, 0)
            labels = torch.squeeze(labels, 1)

            # print(outputs.shape, labels.shape)

            # calculate loss and metrics
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
            spearman_corr_lis.append(corr_s)

            # remove variables from memory
            del inputs, labels, outputs, loss, y_pred_det, y_true_det, corr_p, corr_s
            torch.cuda.empty_cache()
            gc.collect()

        # print(i)

        if (i) % (100) == 0:
            logger.info(f'| samples trained: {(i+1)*bs:5d} | train (intermediate) loss: {total_loss/((i+1)*bs):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')
    
    logger.info(f'Epoch Train Loss: {total_loss/len(train_dataloder): 5.10f} | train pr: {np.mean(pearson_corr_lis):5.10f} | train sr: {np.mean(spearman_corr_lis):5.10f} |')

def validation(model: nn.Module, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs) -> float:
    print("Validating")
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
            inputs, conds, labels, condition = data

            inputs = inputs.long().to(device)

            # process the labels
            labels = labels.float().to(device)
            labels = torch.unsqueeze(labels, 2)

            # shift the labels and send to the decoder
            dec_input = labels[:, :-1]

            # conds
            conds = conds.float().to(device)

            # get predictions
            outputs = model(src = inputs, tgt = dec_input, conds_additional = conds)

            # remove last output token because it is the eos token
            outputs = outputs[:, :-1, :]
            outputs = torch.squeeze(outputs, 0)
            outputs = torch.squeeze(outputs, 1)

            # remove special tokens from labels
            labels = labels[:, 1:-1]
            labels = torch.squeeze(labels, 0)
            labels = torch.squeeze(labels, 1)

            # calculate loss and metrics
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            y_pred_det = outputs.cpu().detach().numpy()
            y_true_det = labels.cpu().detach().numpy()

            # print(y_true_det, y_pred_det)

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            
            corr_lis.append(corr_p)

            # split up correlations by condition
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

def evaluate(model: nn.Module, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs) -> float:
    print("Evaluating")
    model.eval()
    total_loss = 0. 
    corr_lis = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, conds, labels, condition = data
            inputs = inputs.float().to(device)
            print("Sample Number: ", i)
            print("Input Shape: ", inputs.shape)
            # shift inputs to not include the last element
            # inputs = inputs[:, :-1, :]

            # remove first and last element from the labels
            labels = labels.float().to(device)
            num_preds = inputs.shape[1]

            # y_pred
            y_pred = [1e+6]
            y_pred = torch.tensor(y_pred).float().to(device)
            y_pred = torch.unsqueeze(y_pred, 0)
            y_pred = torch.unsqueeze(y_pred, 0)
            # print(y_pred.shape, inputs.shape)

            for i in range(num_preds):
                outputs = model(src = inputs, tgt = y_pred)
                # print(outputs.shape)
                next_pred = outputs[:, -1, :] * 1e+6
                # print(next_pred)
                next_pred = torch.unsqueeze(next_pred, 0)
                y_pred = torch.cat((y_pred, next_pred), dim = 1)

            y_pred = y_pred[:, 1:, :]
            y_pred = torch.squeeze(y_pred, 0)
            y_pred = torch.squeeze(y_pred, 1)

            labels = torch.unsqueeze(labels, 2)
            # labels = labels[:, 1:, :]
            labels = torch.squeeze(labels, 0)
            labels = torch.squeeze(labels, 1)

            # print(y_pred.shape, labels.shape)

            loss = criterion(y_pred, labels)

            total_loss += loss.item()

            y_pred_det = y_pred.cpu().detach().numpy()
            y_true_det = labels.cpu().detach().numpy()

            # print(y_pred_det.shape, y_true_det.shape)

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            print("Y_True: ", y_true_det)
            print("Y_Pred: ", y_pred_det)
            print("Sample PR: ", corr_p)
            
            corr_lis.append(corr_p)
            print("Mean PR: ", np.mean(corr_lis))

    logger.info(f'| PR: {np.mean(corr_lis):5.5f} |')

    return total_loss / (len(test_dataloader) - 1)