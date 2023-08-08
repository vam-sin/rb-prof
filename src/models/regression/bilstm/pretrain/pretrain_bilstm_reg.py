# libraries
import numpy as np
import random
import copy
import time
import torch 
from torch import nn
from model_utils import BiLSTMModel, train, evaluate, RBDataset_NoBS
from os import listdir
from os.path import isfile, join
import sys
import logging
from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef
import argparse
import wandb

# hyperparameters
lr = 1e-4
architecture = 'BiLSTM'
dataset_name = 'MIMO'
feature_list = ['nt', 'cbert', 'conds']
feature_string = '_'.join(feature_list)
loss_func_name = 'PCCLoss'
epochs = 200
bs = 1
saved_files_name = 'B-0-MIMOPretrain-NT_CBERT-BS1-PCCLoss'

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="rb-prof",

    # name the run
    name = saved_files_name,
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": architecture,
    "features": feature_string,
    "dataset": dataset_name,
    "epochs": epochs,
    "batch_size": bs,
    "loss": loss_func_name
    }
)

log_file_name = 'logs/' + saved_files_name + '.log'
model_file_name = 'reg_models/' + saved_files_name + '.pt'

class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcc_l = PearsonCorrCoef().to(torch.device('cuda'))
        
    def forward(self, pred, actual):
        return -1 * self.pcc_l(pred, actual)

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

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            torch.nn.init.xavier_uniform(m.weight)
        if isinstance(m, nn.LSTM):
            torch.nn.init.xavier_uniform(m.weight_ih_l0)
            torch.nn.init.orthogonal(m.weight_hh_l0)
            torch.nn.init.xavier_uniform(m.weight_ih_l0_reverse)
            torch.nn.init.orthogonal(m.weight_hh_l0_reverse)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

if __name__ == '__main__':
    # import data 
    mult_factor = 1
    loss_mult_factor = 1

    # input size
    input_size = 0
    if 'nt' in feature_list:
        input_size += 15
    if 'cbert' in feature_list:
        input_size += 768
    if 'conds' in feature_list:
        input_size += 22
    if 't5' in feature_list:
        input_size += 1024
    if 'lem' in feature_list:
        input_size += 15
    if 'AF2-SS' in feature_list:
        input_size += 5
    if 'mlm_cdna_nt_idai' in feature_list:
        input_size += 7680
    if 'mlm_cdna_nt_pbert' in feature_list:
        input_size += 1536
    if 'cembeds' in feature_list:
        input_size += 256
    if 'geom' in feature_list:
        input_size += 8

    files_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/mimo_data/'

    # pop elements from files_path if they are in genes_to_remov

    print("Starting")

    conditions_list = []

    # train data
    mypath = files_path
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # onlyfiles = [f for f in onlyfiles if f not in genes_to_remove]

    all_files = [mypath + '/' + f for f in onlyfiles]

    # split into train, val, test
    random.shuffle(all_files)

    train_files = all_files[:int(0.8*len(all_files))]
    val_files = all_files[int(0.8*len(all_files)):int(0.9*len(all_files))]
    test_files = all_files[int(0.9*len(all_files)):]

    train_data = RBDataset_NoBS(train_files, dataset_name, feature_list)
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True) 

    print("Loaded Train")

    # val data
    val_data = RBDataset_NoBS(val_files, dataset_name, feature_list)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=True) 

    print("Loaded Val")
    
    # test data
    test_data = RBDataset_NoBS(test_files, dataset_name, feature_list)
    test_dataloader = DataLoader(test_data, bs, shuffle=False) 

    print("Loaded Test")

    logger.info(f'Train Set: {len(train_data):5d} || Validation Set: {len(val_data):5d} || Test Set: {len(test_data): 5d}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    input_dim = input_size
    hidden_dim = 256
    output_dim = 1
    n_layers = 4
    bidirectional = True
    dropout = 0.1
    model = BiLSTMModel(input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    model.apply(init_weights)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    # loss function
    criterion = PCCLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)
    early_stopping_patience = 20
    trigger_times = 0

    best_val_loss = float('inf')
    best_model = None 

    # # CL Experiment
    # model.load_state_dict(torch.load('reg_models/CLExp2_P1-B-0-DS04_More06-NT_CBERT-BS1-PCCLoss.pt'))

    # Training Process
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        logger.info(f'Training Epoch: {epoch:5d}')
        curr_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f'Learning Rate: {curr_lr: 2.10f}')
        wandb.log({"Learning Rate": curr_lr})

        train_loss, train_pr, train_sr =  train(model, train_dataloader, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger)

        wandb.log({"Train Loss": train_loss, "Train Pearson": train_pr})

        logger.info("------------- Validation -------------")
        val_loss, val_pr = evaluate(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.10f}')
        logger.info('-' * 89)

        wandb.log({"Val Loss": val_loss, "Val Pearson": val_pr})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            logger.info("Best Model -- SAVING")
            torch.save(model.state_dict(), model_file_name)
        
        logger.info(f'best val loss: {best_val_loss:5.10f}')

        wandb.log({"Best Val Loss": best_val_loss})

        logger.info("------------- Testing -------------")
        test_loss, test_pr = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'test loss {test_loss:5.10f}')
        logger.info('-' * 89)

        wandb.log({"Test Loss": test_loss, "Test Pearson": test_pr})

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
        val_loss, val_pr = evaluate(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        print('-' * 89)
        print(f'valid loss {val_loss:5.10f}')
        print('-' * 89)

        print("------------- Testing -------------")
        test_loss, test_pr = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        print('-' * 89)
        print(f'test loss {test_loss:5.10f}')
        print('-' * 89)

# wandb finish
wandb.finish()