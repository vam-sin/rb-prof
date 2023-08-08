# libraries
import numpy as np
import random
import copy
import time
import torch 
from torch import nn
from bs_model_utils import TransformerModel, train, evaluate, RBDataset_NoBS
from os import listdir
from os.path import isfile, join
import sys
import logging
from torch.utils.data import DataLoader

saved_files_name = 'E-0-NT_CBERT-CondsCTRLFix_BS16'
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
torch.manual_seed(0)

if __name__ == '__main__':
    # import data 
    mult_factor = 1
    loss_mult_factor = 1
    bs = 1 # batch_size
    train_bs = 16 # batch size in training

    print("Starting")

    # train data
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    train_files = [mypath + '/' + f for f in onlyfiles]

    train_data = RBDataset_NoBS(train_files)
    train_dataloader = DataLoader(train_data, batch_size=train_bs, shuffle=True) 

    print("Loaded Train")
    
    # test data
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    test_files = [mypath + '/' + f for f in onlyfiles]

    test_data = RBDataset_NoBS(test_files)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=True) 

    print("Loaded Test")
    
    # val data
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    val_files = [mypath + '/' + f for f in onlyfiles]

    val_data = RBDataset_NoBS(val_files)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=True) 

    print("Loaded Val")

    logger.info(f'Train Set: {len(train_dataloader):5d} || Validation Set: {len(val_dataloader):5d} || Test Set: {len(test_dataloader): 5d}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    num_feats = 805 # num_feats
    d_hid = 256 # hidden_size
    nlayers = 3 # number of transformer layers
    nhead = 8 # number of attention heads
    dropout = 0.1 # dropout probability
    model = TransformerModel(num_feats, nhead, d_hid, nlayers, mult_factor, dropout).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    criterion = nn.L1Loss()
    lr = 1e-4
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)
    early_stopping_patience = 20
    trigger_times = 0

    best_val_loss = float('inf')
    epochs = 100
    best_model = None 

    # Training Process
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        logger.info(f'Training Epoch: {epoch:5d}')
        curr_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f'Learning Rate: {curr_lr: 2.10f}')
        train(model, train_dataloader, train_bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger)

        logger.info("------------- Validation -------------")
        val_loss = evaluate(model, val_dataloader, device, mult_factor, criterion, logger)
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
        test_loss = evaluate(model, test_dataloader, device, mult_factor, criterion, logger)
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
        val_loss = evaluate(model, val_dataloader, device, mult_factor, criterion, logger)
        print('-' * 89)
        print(f'valid loss {val_loss:5.10f}')
        print('-' * 89)

        print("------------- Testing -------------")
        test_loss = evaluate(model, test_dataloader, device, mult_factor, criterion, logger)
        print('-' * 89)
        print(f'test loss {test_loss:5.10f}')
        print('-' * 89)
