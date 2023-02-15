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
from model_utils import TransformerModel, train, evaluate
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from os import listdir
from os.path import isfile, join
import math
import sys
import logging
import pickle as pkl

saved_files_name = 'TF-Reg-Model-1'
log_file_name = 'logs/' + saved_files_name + '.log'
model_file_name = 'reg_models/' + saved_files_name + '.pt'

# logging setup
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file_name)
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

# reproducibility
random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    # import data 
    mult_factor = 1e+6
    loss_mult_factor = 1

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

    num_feats = 1235
    emsize = 512
    d_hid = 512
    nlayers = 2
    nhead = 2
    dropout = 0.2 
    bs = 64 # batch_size
    model = TransformerModel(num_feats, emsize, nhead, d_hid, nlayers, mult_factor, dropout).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    criterion = nn.MSELoss()
    lr = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)

    best_val_loss = float('inf')
    epochs = 5000
    best_model = None 

    # Training Process
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        logger.info(f'Training Epoch: {epoch:5d}')
        train(model, tr_train, bs, device, criterion, mult_factor, loss_mult_factor)

        logger.info("------------- Validation -------------")
        val_loss = evaluate(model, tr_val, device, mult_factor, criterion)
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.10f}')
        logger.info('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            logger.info("Best Model -- SAVING")
            torch.save(model.state_dict(), model_file_name)
        
        logger.info(f'best val loss: {best_val_loss:5.10f}')

        logger.info("------------- Testing -------------")
        test_loss = evaluate(model, tr_test)
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'test loss {test_loss:5.10f}')
        logger.info('-' * 89)

        scheduler.step(val_loss)

    # Evaluation Metrics
    # model.load_state_dict(torch.load(model_file_name))
    # model.eval()
    # with torch.no_grad():
    #     print("------------- Validation -------------")
    #     val_loss = evaluate(model, tr_val)
    #     print('-' * 89)
    #     print(f'valid loss {val_loss:5.10f}')
    #     print('-' * 89)

    #     print("------------- Testing -------------")
    #     test_loss = evaluate(model, tr_test)
    #     print('-' * 89)
    #     print(f'test loss {test_loss:5.10f}')
    #     print('-' * 89)

'''
MSE Loss: the derivative would be getting really small considering that the values to be predicted are very small and so the mult factor should be very big
MAE Loss could be bigger than the MSE, this is a possible option.
Loss functions:
iXnos and RiboMIMO both use MSE as the loss
ROSE uses NLL (negative log likelihood)
need to find a loss that weights higher differences more and smaller differences, not much (check SmootL1Loss or HingeLoss)
'''

'''
0: MSE Loss: 
1: MSE Loss with the proper division of sequence length
CodonBS_2: MSE Loss with proper div of length and predicts counts per million
CodonBS_3: Huber Loss (delta 0.1) with proper division of length (no mult factors)
CodonBS_4: Huber Loss (delta 0.1) with proper division of length (no mult factors) + (20, 0.6: Dataset Proc) + Added Zeros
CodonBS_5: MSE Loss (1e+6 mult factor) + (20, 0.6: Dataset Proc) + Added Zeros
'''
