# libraries
import numpy as np
from sklearn.model_selection import train_test_split
import random
import gc
import math 
import copy
import time
import torch 
import pandas as pd 
from torch import nn, Tensor 
import torch.nn.functional as F 
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from full_tf_model_utils import TransformerModel, train, evaluate, process_sample
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from os import listdir
from os.path import isfile, join
import math
import sys
import logging
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

# logging setup
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logs/attn_weights.log')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

# reproducibility
random.seed(0)
np.random.seed(0)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        print(output)
        activation[name] = output[0].detach()
    return hook

def attn_matrix_analysis(model: nn.Module, tr_val_key) -> float:
    # print("Evaluating")
    model.eval()
    len_lis = []
    with torch.no_grad():
        # print(i, tr_val[i])
        X, y = process_sample(tr_val_key, mult_factor)
        # print(X.shape)
        seq_len = len(X)
        print(seq_len)
        
        # src_mask = generate_square_subsequent_mask(seq_len).to(device)
        # print(src_mask.shape)
        x_in = torch.from_numpy(X).float().to(device)
        # src_mask = generate_square_subsequent_mask(seq_len).float().to(device)
        y_true = torch.from_numpy(np.expand_dims(y, axis=1)).float().to(device)
        model.transformer.decoder.layers[2].multihead_attn.in_proj_weight.register_forward_hook(get_activation('final_attn'))
        y_pred = torch.squeeze(model(x_in, y_true), dim=1)
        y_true = torch.squeeze(y_true, dim=1)
        # keys = list(model.state_dict().keys())
        # print(keys)
        # for x in range(len(keys)):
        #     print(model.state_dict()[keys[x]].shape, keys[x])
        print(activation['final_attn'].shape)


if __name__ == '__main__':
    # import data 
    mult_factor = 1
    loss_mult_factor = 1e+6

    print("Starting")

    with open('processed_keys/keys_proc_20c_60p.pkl', 'rb') as f:
        onlyfiles = pkl.load(f)

    print(onlyfiles[0])
    logger.info(f'Total Number of Samples: {len(onlyfiles): 4d}')

    logger.info("---- Dataset Processing ----")
    tr_train, tr_test = train_test_split(onlyfiles, test_size=0.2, random_state=42, shuffle=True)
    tr_train, tr_val = train_test_split(tr_train, test_size=0.25, random_state=42, shuffle=True)

    logger.info(f'Train Set: {len(tr_train):5d} || Validation Set: {len(tr_val):5d} || Test Set: {len(tr_test): 5d}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    intoken = 1903 # input features
    outtoken = 1 # output 
    hidden = 256 # hidden layer size
    enc_layers = 3 # number of encoder layers
    dec_layers = 3 # number of decoder layers
    nhead = 8 # number of attention heads
    dropout = 0.1
    bs = 16 # batch_size
    model = TransformerModel(intoken, outtoken, nhead, hidden, enc_layers, dec_layers, dropout).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    criterion = nn.L1Loss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)
    early_stopping_patience = 20
    trigger_times = 0

    best_val_loss = float('inf')
    epochs = 5000
    best_model = None 

    # Evaluation Metrics
    model_file_name = 'reg_models/TF-Reg-Model-FULL_0.pt'
    model.load_state_dict(torch.load(model_file_name))
    model.eval()
    print(model.transformer.decoder.layers[0].self_attn)
    with torch.no_grad():
        print("------------- Getting Attention Matrix -------------")
        k = 4
        for name, _ in model.named_modules():
            print(name)
        print(model._modules["transformer"]._modules["decoder"]._modules["layers"]._modules["2"]._modules["self_attn"].state_dict()['in_proj_weight'].shape)
        attn_matrix_analysis(model, tr_test[k * 10])

'''
'''

'''
'''
