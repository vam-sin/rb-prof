# libraries
import numpy as np
import random
import copy
import time
import torch 
from torch import nn
from model_utils import TransformerModel, validation, RBDataset_NoBS, evaluate, train
from os import listdir
from os.path import isfile, join
import sys
import logging
from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef

# no af2 data
no_af2_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']

saved_files_name = 'TF-0-DS04-CEmbed-CTRLFixConds_SpTokens_BS01-PCCLoss'
log_file_name = 'logs/' + saved_files_name + '.log'
model_file_name = 'reg_models/' + saved_files_name + '.pt'

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

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
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# PCC Loss
class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcc_l = PearsonCorrCoef().to(torch.device('cuda'))
        
    def forward(self, pred, actual):
        return -1 * self.pcc_l(pred, actual)

if __name__ == '__main__':
    # import data 
    mult_factor = 1
    loss_mult_factor = 1
    bs = 1 # batch_size
    dataset_name = 'DS04'
    feature_list = ['nt', 'cbert', 'conds']

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

    input_dim = input_size
    model_dims = 128
    n_heads = 8
    num_enc_layers = 1
    num_dec_layers = 1
    dim_ff = 128
    dropout = 0.1
    model = TransformerModel(input_dim, model_dims, n_heads, num_enc_layers, num_dec_layers, dim_ff, dropout).to(device)
    model.apply(init_weights)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    criterion = PCCLoss()
    criterion = criterion.to(device)

    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        train_loss = train(model, train_dataloader, 1, device, criterion, mult_factor, loss_mult_factor, optimizer, logger)

        logger.info("------------- Val Set -------------")
        val_loss = validation(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
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

        logger.info("------------- Test Set -------------")
        test_loss = validation(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
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
        print("------------- Val Set -------------")
        val_loss = evaluate(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        print('-' * 89)
        print(f'valid loss {val_loss:5.10f}')
        print('-' * 89)

        print("------------- Test Set -------------")
        test_loss = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        print('-' * 89)
        print(f'test loss {test_loss:5.10f}')
        print('-' * 89)