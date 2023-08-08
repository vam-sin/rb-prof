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
from model_utils import CNNModel, train, evaluate, RBDataset_NoBS
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from os import listdir
from os.path import isfile, join
import math
import sys
import logging
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import wandb
from torchmetrics import PearsonCorrCoef

# hyperparameters
lr = 1e-4
architecture = 'CNN'
dataset_name = 'DS06'
feature_list = ['nt', 'cbert', 'conds']
feature_string = '_'.join(feature_list)
loss_func_name = 'PCCLoss'
epochs = 500
bs = 1
saved_files_name = 'CNN-0-DS06-NT_CBERT-BS1-PCCLoss'

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

# no af2 data
no_af2_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)

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
    if 'codon_epa_encodings' in feature_list:
        input_size += 255

    if dataset_name == 'DS06':
        train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
        val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
        test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    elif dataset_name == 'DS04':
        train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/train'
        val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/val'
        test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/test'

    print("Starting")

    # train data
    mypath = train_path
    onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    onlyfiles = []
    for x in onlyfiles_full:
        if x.split('_')[0] not in no_af2_transcripts:
            onlyfiles.append(x)

    train_files = [mypath + '/' + f for f in onlyfiles]

    train_data = RBDataset_NoBS(train_files, dataset_name, feature_list)
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True) 

    print("Loaded Train")

    # val data
    mypath = val_path
    onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    onlyfiles = []
    for x in onlyfiles_full:
        if x.split('_')[0] not in no_af2_transcripts:
            onlyfiles.append(x)

    val_files = [mypath + '/' + f for f in onlyfiles]

    val_data = RBDataset_NoBS(val_files, dataset_name, feature_list)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=True) 

    print("Loaded Val")
    
    # test data
    mypath = test_path
    onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    onlyfiles = []
    for x in onlyfiles_full:
        if x.split('_')[0] not in no_af2_transcripts:
            onlyfiles.append(x)

    test_files = [mypath + '/' + f for f in onlyfiles]

    test_data = RBDataset_NoBS(test_files, dataset_name, feature_list)
    test_dataloader = DataLoader(test_data, bs, shuffle=True) 

    print("Loaded Test")

    logger.info(f'Train Set: {len(train_data):5d} || Validation Set: {len(val_data):5d} || Test Set: {len(test_data): 5d}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    # input_dim = 805
    output_dim = 1
    dropout = 0.1
    model = CNNModel(input_size, output_dim, dropout).to(device)
    model = model.to(torch.float)
    model.apply(init_weights)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    criterion = PCCLoss()
    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 50, factor=0.1, verbose=True)
    early_stopping_patience = 100
    trigger_times = 0

    best_val_loss = float('inf')
    best_model = None 

    # Training Process
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        logger.info(f'Training Epoch: {epoch:5d}')
        curr_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f'Learning Rate: {curr_lr: 2.10f}')
        train_loss, train_pr, train_sr = train(model, train_dataloader, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger)

        wandb.log({"train_loss": train_loss, "train_pr": train_pr, "lr": curr_lr})

        logger.info("------------- Validation -------------")
        val_loss, val_pr = evaluate(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.10f}')
        logger.info('-' * 89)

        wandb.log({"val_loss": val_loss, "val_pr": val_pr})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            logger.info("Best Model -- SAVING")
            torch.save(model.state_dict(), model_file_name)
        
        logger.info(f'best val loss: {best_val_loss:5.10f}')

        wandb.log({"best_val_loss": best_val_loss})

        logger.info("------------- Testing -------------")
        test_loss, test_pr = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'test loss {test_loss:5.10f}')
        logger.info('-' * 89)

        wandb.log({"test_loss": test_loss, "test_pr": test_pr})

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

        wandb.log({"val_loss": val_loss, "val_pr": val_pr, "test_loss": test_loss, "test_pr": test_pr})

'''cnn_0
self.cnn1 = nn.Conv1d(in_channels = input_dim, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
self.cnn2 = nn.Conv1d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
self.cnn3 = nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
self.cnn4 = nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

Test PR: 0.32914
'''

'''cnn_1
self.cnn1 = nn.Conv1d(in_channels = input_dim, out_channels = 512, kernel_size = 5, stride = 1, padding = 2)
self.cnn2 = nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
self.cnn1_2 = nn.Conv1d(in_channels = input_dim, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)

self.cnn3 = nn.Conv1d(in_channels = 256, out_channels = 128, kernel_size = 5, stride = 1, padding = 2)
self.cnn4 = nn.Conv1d(in_channels = 128, out_channels = 64, kernel_size = 5, stride = 1, padding = 2)
self.cnn3_4 = nn.Conv1d(in_channels = 256, out_channels = 64, kernel_size = 5, stride = 1, padding = 2)

Test PR: 0.40383
'''