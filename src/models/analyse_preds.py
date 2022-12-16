'''
- take a random sample
- make predictions using the Transformer Model
- Plot the attention maps and see what is affecting the prediction
'''
# libraries
import pickle as pkl 
import numpy as np
from sklearn.model_selection import train_test_split
import random
import math 
import copy
import time
from typing import Tuple 
import torch 
from torch import nn, Tensor 
import torch.nn.functional as F 
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from tqdm import tqdm

mult_factor = 1e+2

def convertOH(seq, max_len):
    '''
    converts the mRNA sequence into a one-hot encoding 
    P is sequence padding which is [0,0,0,0,0]
    '''
    RNA_dict = {'A': [1,0,0,0,0], 'T': [0,1,0,0,0], 'G': [0,0,1,0,0], 'C': [0,0,0,1,0], 'N': [0,0,0,0,1],'P':[0,0,0,0,0]}
    OH_vec = []
    for i in seq:
        OH_vec.append(RNA_dict[i])
    # for j in range(max_len - len(seq)):
    #     OH_vec.append(RNA_dict['P'])

    OH_vec = np.asarray(OH_vec)

    return OH_vec

def get_maxLen(input_pkl):
    max_len = 0
    dict_keys = list(input_pkl.keys())
    for i in range(len(dict_keys)):
        if max_len < len(input_pkl[dict_keys[i]][0]):
            max_len = len(input_pkl[dict_keys[i]][0])
    
    return max_len

def process_ds(input_pkl, max_len):
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
        seq_vecs.append(np.asarray(convertOH(input_pkl[dict_keys[i]][0], max_len)))
        
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

if __name__ == '__main__':
    device = torch.device('cpu')
    print("Device: ", device)
    num_feats = 5
    emsize = 200
    d_hid = 200 
    nlayers = 2
    nhead = 2
    dropout = 0.2 
    model = TransformerModel(num_feats, emsize, nhead, d_hid, nlayers, dropout).to(device)
    model = model.to(torch.double)
    model.load_state_dict(torch.load('models/tf_simple-perHu_noPad_withMASK.pt'))
    model.eval()
    criterion = nn.MSELoss()

    # import data 
    with open('../../data/rb_prof_Naef/processed_data/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_final.pkl', 'rb') as f:
        dict_seqCounts = pkl.load(f)

    max_len = get_maxLen(dict_seqCounts)
    print("MAX Sequence Length: ", max_len)

    print("---- Dataset Processing ----")
    seq_vecs_train, seq_vecs_val, seq_vecs_test, counts_arrays_train, counts_arrays_val, counts_arrays_test, mask_vecs_train, mask_vecs_val, mask_vecs_test = process_ds(dict_seqCounts, max_len)

    print("Train Set: ", len(seq_vecs_train), "|| Validation Set: ", len(seq_vecs_val), "|| Test Set: " , len(seq_vecs_test))
    print("Sample Shapes from Training Set: ", seq_vecs_train[0].shape, counts_arrays_train[0].shape, mask_vecs_train[0].shape)

    random_sample_num = 36

    with torch.no_grad():
        seq_len = len(seq_vecs_val[random_sample_num])
        src_mask = generate_square_subsequent_mask(seq_len).to(device)
        y_pred = torch.flatten(model(torch.from_numpy(seq_vecs_val[random_sample_num]).double(), src_mask))
        y_true = torch.flatten(torch.from_numpy(counts_arrays_val[random_sample_num]))
        mask_vec_sample = torch.flatten(torch.from_numpy(mask_vecs_val[random_sample_num]))
        loss = criterion(y_pred, y_true)

        y_pred = torch.mul(y_pred, mask_vec_sample).numpy()
        y_true = torch.mul(y_true, mask_vec_sample).numpy()
        mask_vec_sample = mask_vec_sample.numpy()

        print(y_pred.shape)
        print(y_true.shape)
        for j in range(len(y_pred)):
            print(y_pred[j], y_true[j], mask_vec_sample[j])

        print(loss)
        
