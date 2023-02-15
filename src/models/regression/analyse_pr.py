# libraries
import pickle as pkl 
import numpy as np
from sklearn.model_selection import train_test_split
import random
import gc
import math 
import copy
import time
import seaborn as sns
from typing import Tuple 
import torch 
from torch import nn, Tensor 
import torch.nn.functional as F 
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from transformer_reg_fullFT import process_sample, TransformerModel, generate_square_subsequent_mask, PositionalEncoding
from tqdm import tqdm
from scipy.stats import pearsonr
from os import listdir
import os
from os.path import isfile, join
import math
import sys
import logging
import matplotlib.pyplot as plt

# plotting setup
sns.set_style("whitegrid")
sns.set_theme()

# logging setup
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('logs/transformer_analyse.log')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

# reproducibility
random.seed(0)
np.random.seed(0)

mult_factor = 1e+6

def eval_pr(model: nn.Module, tr_val_key) -> float:
    # print("Evaluating")
    model.eval()
    total_loss = 0 
    corr_lis = []
    len_lis = []
    with torch.no_grad():
        # print(i, tr_val[i])
        X, y, mask_vec = process_sample(tr_val_key)
        # print(X.shape)
        seq_len = len(X)
        
        src_mask = generate_square_subsequent_mask(seq_len).to(device)
        # print(src_mask.shape)
        x_in = torch.from_numpy(X).float().to(device)
        y_pred = torch.flatten(model(x_in, src_mask))
        y_true = torch.flatten(torch.from_numpy(y)).to(device)

        loss = criterion(y_pred, y_true)

        y_pred_imp_seq = []
        y_true_imp_seq = []
        for x in range(len(y_pred.cpu().detach().numpy())):
            # if y_true[x].item() != 0:
            y_pred_imp_seq.append(y_pred[x].item())
            y_true_imp_seq.append(y_true[x].item())

        corr, _ = pearsonr(y_true_imp_seq, y_pred_imp_seq) # y axis is predictions, x axis is true value
        logging.info(f'Corr: {corr:3.5f}')
        logging.info(f'Loss: {loss:3.5f}')
        for k in range(len(y_true_imp_seq)):
            len_lis.append(k)
            print(y_true_imp_seq[k], y_pred_imp_seq[k])
        sns.lineplot(x=len_lis, y=y_pred_imp_seq, label='Predicted', color='#f0932b')
        sns.lineplot(x=len_lis, y=y_true_imp_seq, label='True', color='#6ab04c')
        plt.legend()
        plt.show()
        plt.savefig("pr_figs/TF-Reg-Model-0/" + tr_val_key + ".png", format="png")

if __name__ == '__main__':
    # import data 
    with open('keys_proc.pkl', 'rb') as f:
        onlyfiles = pkl.load(f)
    print("Total Number of Samples: ", len(onlyfiles))

    print("---- Dataset Processing ----")
    tr_train, tr_test = train_test_split(onlyfiles, test_size=0.2, random_state=42, shuffle=True)
    tr_train, tr_val = train_test_split(tr_train, test_size=0.25, random_state=42, shuffle=True)

    print("Train Set: ", len(tr_train), "|| Validation Set: ", len(tr_val), "|| Test Set: " , len(tr_test))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    num_feats = 1235
    emsize = 512
    d_hid = 512
    nlayers = 2
    nhead = 2
    dropout = 0.2 
    model = TransformerModel(num_feats, emsize, nhead, d_hid, nlayers, dropout).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(pytorch_total_params)

    criterion = nn.MSELoss()
    lr = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)

    # Evaluation Metrics
    model.load_state_dict(torch.load('reg_models/TF-Reg-Model-0.pt'))
    model.eval()
    with torch.no_grad():
        print("------------- Validation -------------")
        f = 4
        eval_pr(model, tr_train[10 * f])


'''
MSE Loss: the derivative would be getting really small considering that the values to be predicted are very small and so the mult factor should be very big
MAE Loss could be bigger than the MSE, this is a possible option.
'''
