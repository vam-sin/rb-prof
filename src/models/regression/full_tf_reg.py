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
from full_tf_model_utils import TransformerModel, train, evaluate
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from os import listdir
from os.path import isfile, join
import math
import sys
import logging
import pickle as pkl

saved_files_name = 'TF-Reg-Model-FULL_0_small'
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
    mult_factor = 1
    loss_mult_factor = 1e+6

    print("Starting")

    with open('processed_keys/keys_proc_20c_60p.pkl', 'rb') as f:
        onlyfiles = pkl.load(f)

    print(onlyfiles[0])
    logger.info(f'Total Number of Samples: {len(onlyfiles): 4d}')

    logger.info("---- Dataset Processing ----")
    tr_train, tr_test = train_test_split(onlyfiles, test_size=0.2, random_state=42, shuffle=True)
    # tr_train, tr_val = train_test_split(tr_train, test_size=0.25, random_state=42, shuffle=True)
    tr_train, tr_val = train_test_split(tr_train, test_size=0.9, random_state=42, shuffle=True)

    logger.info(f'Train Set: {len(tr_train):5d} || Validation Set: {len(tr_val):5d} || Test Set: {len(tr_test): 5d}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    intoken = 96 # input features
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

    # # Training Process
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        logger.info(f'Training Epoch: {epoch:5d}')
        curr_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f'Learning Rate: {curr_lr: 2.10f}')
        train(model, tr_train, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger)

        logger.info("------------- Validation -------------")
        val_loss = evaluate(model, tr_val, device, mult_factor, criterion, logger)
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
        test_loss = evaluate(model, tr_test, device, mult_factor, criterion, logger)
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'test loss {test_loss:5.10f}')
        logger.info('-' * 89)

        scheduler.step(val_loss)

        # early stopping criterion
        if val_loss > best_val_loss:
          trigger_times += 1
          logger.info(f'| Trigger Times: {trigger_times:4d} |')
          if trigger_times >= early_stopping_patience:
            logger.info('------------- Early Stoppping -------------')
            break 
        else:
          trigger_times = 0
          logger.info(f'| Trigger Times: {trigger_times:4d} |')

    # Evaluation Metrics
    model.load_state_dict(torch.load(model_file_name))
    model.eval()
    with torch.no_grad():
        print("------------- Validation -------------")
        val_loss = evaluate(model, tr_val, device, mult_factor, criterion, logger)
        print('-' * 89)
        print(f'valid loss {val_loss:5.10f}')
        print('-' * 89)

        print("------------- Testing -------------")
        test_loss = evaluate(model, tr_test, device, mult_factor, criterion, logger)
        print('-' * 89)
        print(f'test loss {test_loss:5.10f}')
        print('-' * 89)

'''
'''

'''
'''
