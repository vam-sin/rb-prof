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
from model_utils import BiLSTMModel, train, evaluate, RBDataset, RBDataset_NoBS
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from os import listdir
from os.path import isfile, join
import math
import sys
import logging
import pickle as pkl
from torch.utils.data import Dataset, DataLoader

saved_files_name = 'B-0-DS06-NT_CBERT-BS1-PCCLoss'
log_file_name = 'logs/' + saved_files_name + '.log'
model_file_name = 'reg_models/' + saved_files_name + '.pt'
# more_model_file_name = 'reg_models/' + saved_files_name + '_MORE.pt'

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

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)

if __name__ == '__main__':
    # import data 
    mult_factor = 1
    loss_mult_factor = 1
    bs = 1 # batch_size

    print("Starting")

    # train data
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    train_files = [mypath + '/' + f for f in onlyfiles]

    # remove files with ILE
    # train_files = []
    # for f in train_files_full:
    #     if '_CTRL_' in f or '_LEU_' in f or '_VAL_' in f:
    #         train_files.append(f)

    train_data = RBDataset_NoBS(train_files)
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True) 
    # X, y, l = next(iter(train_dataloader))

    print("Loaded Train")

    # val data
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    val_files = [mypath + '/' + f for f in onlyfiles]

    # remove files with ILE
    # val_files = []
    # for f in val_files_full:
    #     if '_CTRL_' in f or '_LEU_' in f or '_VAL_' in f:
    #         val_files.append(f)

    val_data = RBDataset_NoBS(val_files)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=True) 

    print("Loaded Val")
    
    # test data
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    test_files = [mypath + '/' + f for f in onlyfiles]

    # remove files with ILE
    # test_files = []
    # for f in test_files_full:
    #     if '_CTRL_' in f or '_LEU_' in f or '_VAL_' in f:
    #         test_files.append(f)

    test_data = RBDataset_NoBS(test_files)
    test_dataloader = DataLoader(test_data, bs, shuffle=True) 

    print("Loaded Test")

    logger.info(f'Train Set: {len(train_data):5d} || Validation Set: {len(val_data):5d} || Test Set: {len(test_data): 5d}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    input_dim = 805
    embedding_dim = 64
    hidden_dim = 256
    output_dim = 1
    n_layers = 4
    bidirectional = True
    dropout = 0.1
    model = BiLSTMModel(input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    model = model.to(torch.float)
    # model.apply(init_weights)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    criterion = nn.L1Loss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 50, factor=0.1, verbose=True)
    early_stopping_patience = 100
    trigger_times = 0

    best_val_loss = float('inf')
    epochs = 5000
    best_model = None 
    # # model.load_state_dict(torch.load(model_file_name))

    # # Training Process
    # for epoch in range(1, epochs + 1):
    #     epoch_start_time = time.time()
        
    #     logger.info(f'Training Epoch: {epoch:5d}')
    #     curr_lr = scheduler.optimizer.param_groups[0]['lr']
    #     logger.info(f'Learning Rate: {curr_lr: 2.10f}')
    #     train(model, train_dataloader, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger)

    #     logger.info("------------- Validation -------------")
    #     val_loss = evaluate(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
    #     elapsed = time.time() - epoch_start_time
    #     logger.info('-' * 89)
    #     logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
    #       f'valid loss {val_loss:5.10f}')
    #     logger.info('-' * 89)

    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         best_model = copy.deepcopy(model)
    #         logger.info("Best Model -- SAVING")
    #         torch.save(model.state_dict(), model_file_name)
        
    #     logger.info(f'best val loss: {best_val_loss:5.10f}')

    #     logger.info("------------- Testing -------------")
    #     test_loss = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
    #     elapsed = time.time() - epoch_start_time
    #     logger.info('-' * 89)
    #     logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
    #       f'test loss {test_loss:5.10f}')
    #     logger.info('-' * 89)

    #     scheduler.step(val_loss)

    #     # early stopping criterion
    #     if val_loss > best_val_loss:
    #       trigger_times += 1
    #       logger.info(f'| Trigger Times: {trigger_times:4d} |')
    #       if trigger_times >= early_stopping_patience:
    #         logger.info('------------- Early Stoppping -------------')
    #         break 
    #     else:
    #       trigger_times = 0
    #       logger.info(f'| Trigger Times: {trigger_times:4d} |')

    # Evaluation Metrics
    model.load_state_dict(torch.load(model_file_name))
    model.eval()
    with torch.no_grad():
        # print("------------- Validation -------------")
        # val_loss = evaluate(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        # print('-' * 89)
        # print(f'valid loss {val_loss:5.10f}')
        # print('-' * 89)

        print("------------- Testing -------------")
        test_loss = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        print('-' * 89)
        print(f'test loss {test_loss:5.10f}')
        print('-' * 89)

