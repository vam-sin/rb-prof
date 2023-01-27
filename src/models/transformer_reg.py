# libraries
import pickle as pkl 
import numpy as np
from sklearn.model_selection import train_test_split
import random
import gc
import math 
import copy
import time
from typing import Tuple 
import torch 
from torch import nn, Tensor 
import torch.nn.functional as F 
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from tqdm import tqdm
from scipy.stats import pearsonr

# reproducibility
random.seed(0)
np.random.seed(0)

mult_factor = 1e+2

'''
'-' is for those beginning amino acids for which the codon cant be defined. 
'''

codon_embeds = np.load('../../data/rb_prof_Naef/processed_data_full/codons_embeds.npy', allow_pickle='TRUE').item()

def convertOH(seq):
    '''
    converts the mRNA sequence into a one-hot encoding 
    '''
    RNA_dict = {'A': [1,0,0,0,0], 'T': [0,1,0,0,0], 'G': [0,0,1,0,0], 'C': [0,0,0,1,0], 'N': [0,0,0,0,1]}
    OH_vec = []
    codon_vec = []
    for i in seq:
        OH_vec.append(RNA_dict[i])
    
    for i in range(len(seq)):
        # print(i)
        if i%3 == 0:
            codon_seq = seq[i:i+3]
        if i%3 == 1:
            codon_seq = seq[i-1:i+2]
        if i%3 == 2:
            codon_seq = seq[i-2:i+1]
        
        if len(codon_seq) < 3:
            codon_vec.append(codon_embeds['-'])
        else:
            codon_vec.append(codon_embeds[str(codon_seq)])

    OH_vec = np.asarray(OH_vec)
    codon_vec = np.asarray(codon_vec)
    append_vec = np.concatenate((OH_vec, codon_vec), axis=1)

    return append_vec

def get_maxLen(input_pkl):
    max_len = 0
    dict_keys = list(input_pkl.keys())
    for i in range(len(dict_keys)):
        if max_len < len(input_pkl[dict_keys[i]][0]):
            max_len = len(input_pkl[dict_keys[i]][0])
    
    return max_len

def process_ds(input_pkl):
    '''
    conducts the whole preprocessing of the input pickle file:
    1. converts sequences to onehot
    2. pads the sequences so that everything is in the same shape: padding for output is '1'
    3. splits the data into train-test-val
    '''
    dict_keys = list(input_pkl.keys())
    seq_vecs = []
    counts_arrays = []
    mask_vecs = []
    for i in range(len(dict_keys)):
        # sequence vectors
        if len(input_pkl[dict_keys[i]][0]) < 10000:
            seq_vecs.append(np.asarray(convertOH(input_pkl[dict_keys[i]][0])))
            
            # count vectors
            # counts per million
            c_arr_sample = np.asarray(input_pkl[dict_keys[i]][1]) * mult_factor
            
            # c_arr_sample = np.asarray(input_pkl[dict_keys[i]][1])
            # full_c_arr_sample = np.ones((max_len,)) # extra ones are padding
            # full_c_arr_sample[0:len(c_arr_sample)] = c_arr_sample
            # full_c_arr_sample = np.expand_dims(full_c_arr_sample, axis=1)
            counts_arrays.append(c_arr_sample)

            # mask vectors 
            # full_mask_sample = np.zeros((max_len,))
            mask_sample = []
            for j in range(len(c_arr_sample)):
                if c_arr_sample[j] == 0.0:
                    mask_sample.append(0)
                else:
                    mask_sample.append(1)
            mask_sample = np.asarray(mask_sample)
            # full_mask_sample[0:len(c_arr_sample)] = mask_sample
            mask_vecs.append(np.expand_dims(mask_sample, axis=1))

    # counts_arrays = np.asarray(counts_arrays)
    # seq_vecs = np.asarray(seq_vecs)
    # # codon_vecs_full = np.asarray(codon_vecs_full)
    # print(seq_vecs.shape)
    # print(seq_vecs.shape, codon_vecs_full.shape)
    # mask_vecs = np.asarray(mask_vecs)

    seq_vecs_train, seq_vecs_test, counts_arrays_train, counts_arrays_test, mask_vecs_train, mask_vecs_test = train_test_split(seq_vecs, counts_arrays, mask_vecs, test_size=0.2, random_state=42, shuffle=True)
    seq_vecs_train, seq_vecs_val, counts_arrays_train, counts_arrays_val, mask_vecs_train, mask_vecs_val = train_test_split(seq_vecs_train, counts_arrays_train, mask_vecs_train, test_size=0.25, random_state=42, shuffle=True)

    return seq_vecs_train, seq_vecs_val, seq_vecs_test, counts_arrays_train, counts_arrays_val, counts_arrays_test, mask_vecs_train, mask_vecs_val, mask_vecs_test

class TransformerModel(nn.Module):
    def __init__(self, num_feats: int, d_model: int, nhead: int, d_hid: int,
                nlayers: int, dropout: float = 0.5):
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
        self.d_model = d_model 
        self.decoder = nn.Linear(d_model, 1)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1 
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        # print(src.shape)
        # print(self.encoder(src).shape)
        src = self.encoder(src) * math.sqrt(self.d_model)
        # print(src.shape)
        src = self.pos_encoder(src)
        output = self.relu(self.transformer_encoder(src, src_mask))
        # print(output.shape)
        output = self.decoder(output)
        # print(output.shape)
 
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
        # print(x.shape, self.pe[:x.size(0)].shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def train(model: nn.Module, seq_vecs_train, counts_arrays_train, mask_vecs_train) -> None:
    model.train()
    total_loss = 0. 
    log_interval = 100 
    start_time = time.time() 
    corr_lis = []
    for i in range(len(seq_vecs_train)):
        seq_len = len(seq_vecs_train[i])
        x_in = torch.from_numpy(seq_vecs_train[i]).float().to(device)
        src_mask = generate_square_subsequent_mask(seq_len).float().to(device)
        y_pred = torch.flatten(model(x_in, src_mask))
        y_true = torch.flatten(torch.from_numpy(counts_arrays_train[i])).float().to(device)
        mask_vec_sample = torch.flatten(torch.from_numpy(mask_vecs_train[i])).to(device)

        y_pred = torch.mul(y_pred, mask_vec_sample)
        y_true = torch.mul(y_true, mask_vec_sample)

        loss = criterion(y_pred, y_true)

        y_pred_imp_seq = []
        y_true_imp_seq = []
        for x in range(len(y_pred.cpu().detach().numpy())):
            if y_true[x].item() != 0:
                y_pred_imp_seq.append(y_pred[x].item())
                y_true_imp_seq.append(y_true[x].item())
        # print(y_pred_imp_seq)
        corr, _ = pearsonr(y_true_imp_seq, y_pred_imp_seq)
        print(corr)
        if corr == np.nan:
            print(corr, y_pred_imp_seq)

        corr_lis.append(corr)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        # remove from GPU device
        del x_in
        del src_mask
        del y_true
        del mask_vec_sample
        del y_pred
        gc.collect()
        torch.cuda.empty_cache()

        if (i+1) % log_interval == 0:
            print(f'| samples trained: {i+1:5d} | train (intermediate) loss: {total_loss/(i+1):5.10f} | train (intermediate) pr: {np.mean(corr_lis):5.10f} |')

    print(f'Epoch Train Loss: {total_loss/len(seq_vecs_train): 5.10f} | train pr: {np.mean(corr_lis):5.10f} |')

def evaluate(model: nn.Module, seq_vecs_val, counts_arrays_val, mask_vecs_val) -> float:
    # print("Evaluating")
    model.eval()
    total_loss = 0 
    corr_lis = []
    with torch.no_grad():
        for i in range(len(seq_vecs_val)):
            seq_len = len(seq_vecs_val[i])
            src_mask = generate_square_subsequent_mask(seq_len).to(device)
            x_in = torch.from_numpy(seq_vecs_val[i]).float().to(device)
            y_pred = torch.flatten(model(x_in, src_mask))
            y_true = torch.flatten(torch.from_numpy(counts_arrays_val[i])).to(device)
            mask_vec_sample = torch.flatten(torch.from_numpy(mask_vecs_val[i])).to(device)

            y_pred = torch.mul(y_pred, mask_vec_sample)
            y_true = torch.mul(y_true, mask_vec_sample)

            loss = criterion(y_pred, y_true)

            y_pred_imp_seq = []
            y_true_imp_seq = []
            for x in range(len(y_pred.cpu().detach().numpy())):
                if y_true[x].item() != 0:
                    y_pred_imp_seq.append(y_pred[x].item())
                    y_true_imp_seq.append(y_true[x].item())

            corr, _ = pearsonr(y_true_imp_seq, y_pred_imp_seq)
            corr_lis.append(corr)

            total_loss += loss.item()

            del x_in
            del src_mask
            del y_pred
            del y_true
            del mask_vec_sample
            del loss
            
            gc.collect()
            torch.cuda.empty_cache()

    print(f'| PR: {np.mean(corr_lis):5.5f} |')

    return total_loss / (len(seq_vecs_val) - 1)

if __name__ == '__main__':
    # import data 
    with open('../../data/rb_prof_Naef/processed_proper/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_finalNonEmpty.pkl', 'rb') as f:
        dict_seqCounts = pkl.load(f)

    max_len = get_maxLen(dict_seqCounts)
    print("MAX Sequence Length: ", max_len)

    print("---- Dataset Processing ----")
    seq_vecs_train, seq_vecs_val, seq_vecs_test, counts_arrays_train, counts_arrays_val, counts_arrays_test, mask_vecs_train, mask_vecs_val, mask_vecs_test = process_ds(dict_seqCounts)

    print("Train Set: ", len(seq_vecs_train), "|| Validation Set: ", len(seq_vecs_val), "|| Test Set: " , len(seq_vecs_test))
    print("Sample Shapes from Training Set: ", seq_vecs_train[0].shape, counts_arrays_train[0].shape, mask_vecs_train[0].shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    num_feats = 105
    emsize = 512
    d_hid = 2048
    nlayers = 2
    nhead = 2
    dropout = 0.2 
    model = TransformerModel(num_feats, emsize, nhead, d_hid, nlayers, dropout).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    criterion = nn.MSELoss()
    lr = 1e-6
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)

    best_val_loss = float('inf')
    epochs = 5000
    best_model = None 

    # Training Process
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        print(f'Training Epoch: {epoch:5d}')
        train(model, seq_vecs_train, counts_arrays_train, mask_vecs_train)

        print("------------- Validation -------------")
        val_loss = evaluate(model, seq_vecs_val, counts_arrays_val, mask_vecs_val)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.10f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            print("Best Model -- SAVING")
            torch.save(model.state_dict(), 'models/TF_Reg_0.pt')
        
        print(f'best val loss: {best_val_loss:5.10f}')

        print("------------- Testing -------------")
        test_loss = evaluate(model, seq_vecs_test, counts_arrays_test, mask_vecs_test)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'test loss {test_loss:5.10f}')
        print('-' * 89)

        scheduler.step(val_loss)

    # Evaluation Metrics
    model.load_state_dict(torch.load('models/TF_Reg_0.pt'))
    model.eval()
    with torch.no_grad():
        print("------------- Validation -------------")
        val_loss = evaluate(model, seq_vecs_val, counts_arrays_val, mask_vecs_val)
        print('-' * 89)
        print(f'valid loss {val_loss:5.10f}')
        print('-' * 89)

        print("------------- Testing -------------")
        test_loss = evaluate(model, seq_vecs_test, counts_arrays_test, mask_vecs_test)
        print('-' * 89)
        print(f'test loss {test_loss:5.10f}')
        print('-' * 89)


'''
the inital model (tf_simple_noPad): (unsure if this had any non-linearity) was predicting the same value for everything and was not learning anything. (predicting negative value)
added non-linearity ReLU: 
'''

'''MSE Loss
------------- Validation -------------
-----------------------------------------------------------------------------------------
valid loss 0.0056834715
-----------------------------------------------------------------------------------------
------------- Testing -------------
-----------------------------------------------------------------------------------------
test loss 0.0045646601
'1.12.1+cu102'
'''
