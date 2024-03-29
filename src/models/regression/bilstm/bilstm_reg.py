# libraries
import numpy as np
import random
import copy
import time
import torch 
from torch import nn
from model_utils import BiLSTMModel, train, evaluate, RBDataset_NoBS, PCCLoss, PoissonLoss, CombinedPoissonPCCLoss
from os import listdir
from os.path import isfile, join
import sys
import logging
from torch.utils.data import DataLoader
import wandb

# hyperparameters
lr = 1e-5
weight_decay = 0.01
architecture = 'BiLSTM-0'
dataset_name = 'DS06'
feature_list = ['cembeds', 'conds']
feature_string = '_'.join(feature_list)
loss_func_name = 'PCCLoss'
epochs = 100
bs = 1
dropout = 0.1
condition = 'noNnt'

saved_files_name = architecture + '-' + dataset_name + '-' + feature_string + '-' + loss_func_name + '-BS' + str(bs) + '-LR_' + str(lr) + '-E_' + str(epochs) + '-LogNorm' + '-Dropout_' + str(dropout) + '-' + condition

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="rb-prof",

    # name the run
    name = saved_files_name,
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "architecture": architecture,
    "features": feature_string,
    "dataset": dataset_name,
    "epochs": epochs,
    "batch_size": bs,
    "loss": loss_func_name,
    "dropout": dropout
    }
)

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

# no af2 data
no_af2_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']

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
    if 'codon_epa_encodings' in feature_list:
        input_size += 255
    if 'svdimred' in feature_list:
        input_size += 64

    if dataset_name == 'DS06':
        train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
        val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
        test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    elif dataset_name == 'DS06_Liver06':
        liver_files_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/liver_06'
        train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
        val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
        test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    elif dataset_name == 'Liver06':
        liver_files_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/liver_06'
    elif dataset_name == 'DS04':
        train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/train'
        val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/val'
        test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/test'

    print("Starting")

    conditions_list = []

    # train data
    mypath = train_path
    onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    onlyfiles = []
    for x in onlyfiles_full:
        if x.split('_')[0] not in no_af2_transcripts:
            onlyfiles.append(x)

    train_files = [mypath + '/' + f for f in onlyfiles]

    # add liver data
    if dataset_name == 'DS06_Liver06':
        onlyfiles_liver = [f for f in listdir(liver_files_path) if isfile(join(liver_files_path, f))]
        onlyfiles_full_liver = [liver_files_path + '/' + f for f in onlyfiles_liver]

        full_training_files = [] 
        for x in train_files:
            full_training_files.append(x)
        for x in onlyfiles_full_liver:
            full_training_files.append(x)

        # shuffle train files
        random.shuffle(full_training_files)

        train_data = RBDataset_NoBS(full_training_files, dataset_name, feature_list)
        
    else:
        train_data = RBDataset_NoBS(train_files, dataset_name, feature_list)

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

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
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True) 

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
    test_dataloader = DataLoader(test_data, bs, shuffle=False, num_workers=4, pin_memory=True) 

    print("Loaded Test")

    logger.info(f'Train Set: {len(train_data):5d} || Validation Set: {len(val_data):5d} || Test Set: {len(test_data): 5d}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    # bilstm model building
    input_dim = input_size
    if architecture == 'BiLSTM-0':
        hidden_dim = 256
        output_dim = 1
        n_layers = 4
        bidirectional = True
    model = BiLSTMModel(input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    # model.apply(init_weights)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    # loss function
    if loss_func_name == 'CombinedPoissonPCCLoss':
        criterion = CombinedPoissonPCCLoss().to(device)
    elif loss_func_name == 'PCCLoss':
        criterion = PCCLoss().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 25, factor=0.1, verbose=True)
    early_stopping_patience = 50
    trigger_times = 0

    best_val_loss = float('inf')
    best_model = None 

    # Training Process
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        logger.info(f'Training Epoch: {epoch:5d}')
        curr_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f'Learning Rate: {curr_lr: 2.10f}')
        wandb.log({"Learning Rate": curr_lr})

        train_loss, train_pr, train_sr =  train(model, train_dataloader, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger, epoch, epochs)

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
