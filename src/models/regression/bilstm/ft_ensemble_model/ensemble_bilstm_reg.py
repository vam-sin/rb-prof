# libraries
import numpy as np
import random
import copy
import time
import torch 
from torch import nn
from ensemble_model_utils import BiLSTMModel, train, evaluate, RBDataset_NoBS, LinearRegressionModel
from os import listdir
from os.path import isfile, join
import sys
import logging
from torch.utils.data import DataLoader
from torchmetrics import PearsonCorrCoef
import argparse

saved_files_name = 'EB-0-DS06-ALL-BS1-PCCLoss'
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
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    # import data 
    mult_factor = 1
    loss_mult_factor = 1
    bs = 1 # batch_size
    dataset_name = 'DS06'
    feature_list = ['AF2-SS',  'conds']

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

    if dataset_name == 'DS06':
        train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
        val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
        test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    elif dataset_name == 'DS04':
        train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/train'
        val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/val'
        test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/test'

    # define template model
    hidden_dim = 256
    output_dim = 1
    n_layers = 4
    bidirectional = True
    dropout = 0.1

    # load different feature models
    # nt model
    nt_model = BiLSTMModel(37, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    nt_model = nt_model.to(torch.float)
    nt_model.load_state_dict(torch.load('../reg_models/B-0-DS06-NT-BS1-PCCLoss.pt'))
    nt_model.eval()
    
    # cbert model
    cbert_model = BiLSTMModel(790, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    cbert_model = cbert_model.to(torch.float)
    cbert_model.load_state_dict(torch.load('../reg_models/B-0-DS06-CBERT-BS1-PCCLoss.pt'))
    cbert_model.eval()

    # t5 model
    t5_model = BiLSTMModel(1046, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    t5_model = t5_model.to(torch.float)
    t5_model.load_state_dict(torch.load('../reg_models/B-0-DS06-T5-BS1-PCCLoss.pt'))
    t5_model.eval()

    # lem model
    lem_model = BiLSTMModel(37, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    lem_model = lem_model.to(torch.float)
    lem_model.load_state_dict(torch.load('../reg_models/B-0-DS06-LEM-BS1-PCCLoss.pt'))
    lem_model.eval()

    # AF2-SS model
    af2_ss_model = BiLSTMModel(27, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    af2_ss_model = af2_ss_model.to(torch.float)
    af2_ss_model.load_state_dict(torch.load('../reg_models/B-0-DS06-AF2-BS1-PCCLoss.pt'))
    af2_ss_model.eval()

    # mlm_cdna_nt_idai model
    mlm_cdna_nt_idai_model = BiLSTMModel(7702, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    mlm_cdna_nt_idai_model = mlm_cdna_nt_idai_model.to(torch.float)
    mlm_cdna_nt_idai_model.load_state_dict(torch.load('../reg_models/B-0-DS06-mlm_IDAI-BS1-PCCLoss.pt'))
    mlm_cdna_nt_idai_model.eval()

    # mlm_cdna_nt_pbert model
    mlm_cdna_nt_pbert_model = BiLSTMModel(1558, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    mlm_cdna_nt_pbert_model = mlm_cdna_nt_pbert_model.to(torch.float)
    mlm_cdna_nt_pbert_model.load_state_dict(torch.load('../reg_models/B-0-DS06-mlm_PBERT-BS1-PCCLoss.pt'))
    mlm_cdna_nt_pbert_model.eval()

    # geom model
    geom_model = BiLSTMModel(30, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    geom_model = geom_model.to(torch.float)
    geom_model.load_state_dict(torch.load('../reg_models/B-0-DS06-Geom-BS1-PCCLoss.pt'))
    geom_model.eval()

    # cembeds model
    # cembeds_model = BiLSTMModel_Embedding(87, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    # cembeds_model = cembeds_model.to(torch.float)
    # cembeds_model.load_state_dict(torch.load('../reg_models/B-0-DS06-CEmbeds-BS1-PCCLoss.pt'))
    # cembeds_model.eval()

    # models dict
    models_dict = {'nt': nt_model, 'cbert': cbert_model, 't5': t5_model, 'lem': lem_model, 'af2_ss': af2_ss_model, 'mlm_cdna_nt_idai': mlm_cdna_nt_idai_model, 'mlm_cdna_nt_pbert': mlm_cdna_nt_pbert_model, 'geom': geom_model}

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

    train_data = RBDataset_NoBS(train_files, dataset_name, feature_list, models_dict)
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

    val_data = RBDataset_NoBS(val_files, dataset_name, feature_list, models_dict)
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

    test_data = RBDataset_NoBS(test_files, dataset_name, feature_list, models_dict)
    test_dataloader = DataLoader(test_data, bs, shuffle=False) 

    print("Loaded Test")

    logger.info(f'Train Set: {len(train_data):5d} || Validation Set: {len(val_data):5d} || Test Set: {len(test_data): 5d}')

    # ensemble lin reg model
    input_dim = 512
    output_dim = 1
    dropout = 0.5
    model = LinearRegressionModel(input_dim, output_dim, dropout).to(device)
    # model.apply(init_weights)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    # loss function
    criterion = PCCLoss().to(device)

    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)
    early_stopping_patience = 20
    trigger_times = 0

    best_val_loss = float('inf')
    epochs = 200
    best_model = None 

    # Training Process
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        logger.info(f'Training Epoch: {epoch:5d}')
        curr_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f'Learning Rate: {curr_lr: 2.10f}')

        train(model, train_dataloader, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger)

        logger.info("------------- Validation -------------")
        val_loss = evaluate(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
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
        test_loss = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
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
        val_loss = evaluate(model, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        print('-' * 89)
        print(f'valid loss {val_loss:5.10f}')
        print('-' * 89)

        print("------------- Testing -------------")
        test_loss = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs)
        print('-' * 89)
        print(f'test loss {test_loss:5.10f}')
        print('-' * 89)