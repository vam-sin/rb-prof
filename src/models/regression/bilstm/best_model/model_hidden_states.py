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
from model_utils import BiLSTMModel, train, evaluate, RBDataset, RBDataset_NoBS, hidden_states_eval, BiLSTMModel_Explain
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from os import listdir
from os.path import isfile, join
import math
import sys
import logging
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score

saved_files_name = 'B-0-HS'
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

    print("Starting")
    
    # test data
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    test_files = [mypath + '/' + f for f in onlyfiles]

    test_data = RBDataset_NoBS(test_files)
    test_dataloader = DataLoader(test_data, bs, shuffle=True) 

    print("Loaded Test")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    input_dim = 805
    embedding_dim = 64
    hidden_dim = 256
    output_dim = 1
    n_layers = 4
    bidirectional = True
    dropout = 0.1
    model = BiLSTMModel_Explain(input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Params Total: {pytorch_total_params: 4d}')

    criterion = nn.L1Loss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 50, factor=0.1, verbose=True)

    model.load_state_dict(torch.load('reg_models/B-0-DS06-NT_CBERT-BS1-PCCLoss.pt'))
    model.eval()
    with torch.no_grad():
        print("------------- Testing -------------")
        hidden_states_eval(model, test_dataloader, device, logger)

    # load npy files
    file_path = ''
    hidden_states = np.load(file_path + 'hidden_states.npy')
    labels = np.load(file_path + 'conds.npy')
    cell_states = np.load(file_path + 'cell_states.npy')

    # clustering Hidden States
    kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(hidden_states)
    print(kmeans)
    # save k means
    np.save(file_path + 'kmeans.npy', kmeans)
    # print unique elements in ground truth labels 
    print(labels)

    # get performance of clustering given the ground truth labels and the predicted labels
    ri = rand_score(labels, kmeans)
    ari = adjusted_rand_score(labels, kmeans)
    mi = mutual_info_score(labels, kmeans)
    nmi = normalized_mutual_info_score(labels, kmeans)
    ami = adjusted_mutual_info_score(labels, kmeans)
    print("Rand Index Hidden: ", ri)
    print("Adjusted Rand Index Hidden: ", ari)
    print("Mutual Info Hidden: ", mi)
    print("Normalized Mutual Info Hidden: ", nmi)
    print("Adjusted Mutual Info Hidden: ", ami)

    # clustering Cell States
    kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(cell_states)
    print(kmeans)
    # print unique elements in ground truth labels 
    print(labels)

    # get performance of clustering given the ground truth labels and the predicted labels
    ri = rand_score(labels, kmeans)
    ari = adjusted_rand_score(labels, kmeans)
    mi = mutual_info_score(labels, kmeans)
    nmi = normalized_mutual_info_score(labels, kmeans)
    ami = adjusted_mutual_info_score(labels, kmeans)
    print("Rand Index Cell: ", ri)
    print("Adjusted Rand Index Cell: ", ari)
    print("Mutual Info Cell: ", mi)
    print("Normalized Mutual Info Cell: ", nmi)
    print("Adjusted Mutual Info Cell: ", ami)

'''Clustering does not look good, it is basically random
Rand Index Hidden:  0.6461521191454592
Adjusted Rand Index Hidden:  -0.0017103184379740212
Mutual Info Hidden:  0.007812722851670015
Normalized Mutual Info Hidden:  0.0049598936100028615
Adjusted Mutual Info Hidden:  -0.0032603510980572857

Rand Index Cell:  0.6530181099478656
Adjusted Rand Index Cell:  0.02005429937458287
Mutual Info Cell:  0.03865497491194504
Normalized Mutual Info Cell:  0.024533874197871043
Adjusted Mutual Info Cell:  0.016452690956201235
'''