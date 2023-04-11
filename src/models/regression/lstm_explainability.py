# libraries
import pickle as pkl 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import torch 
from torch import nn
from full_ds_bilstm_model_utils import RBDataset_NoBS, BiLSTMModel
from scipy.stats import pearsonr
from os import listdir
from os.path import isfile, join
import sys
import logging
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import lime 
import lime.lime_tabular

# plotting setup
sns.set_style("ticks")
# sns.set_theme()

# logging setup
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('logs/transformer_analyse.log')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

# reproducibility
random.seed(0)
np.random.seed(0)

mult_factor = 1

if __name__ == '__main__':
    # import data 
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    test_files = [mypath + '/' + f for f in onlyfiles]

    test_data = RBDataset_NoBS(test_files)
    bs = 1
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=True) 
    print("Total Number of Test Samples: ", len(test_dataloader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    input_dim = 803
    embedding_dim = 64
    hidden_dim = 256
    output_dim = 1
    n_layers = 4
    bidirectional = True
    dropout = 0.2
    model = BiLSTMModel(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(pytorch_total_params)

    criterion = nn.L1Loss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)

    # Evaluation Metrics
    model.load_state_dict(torch.load('reg_models/FULLDS_BiLSTM-Reg-3.pt'))
    model.eval()
    with torch.no_grad():
        # print(i, tr_val[i])
        inputs, labels = next(iter(test_dataloader))
        inputs = inputs.float().to('cpu').numpy()
        inputs = inputs[0,:]

        ft_names = ['reg_out']
        # for j in range(15):
        #     ft_names.append('nt_' + str(j+1))
        # for j in range(803-15):
        #     ft_names.append('cb_' + str(j+1))
        print(len(ft_names), inputs.shape, labels)
        explainer = lime.lime_tabular.LimeTabularExplainer(inputs, feature_names=ft_names, verbose=True, mode='regression')
        exp = explainer.explain_instance(inputs, model.predict, num_features=803)
        # exp.show_in_notebook(show_table=True, show_all=False)
        print(exp.as_list())

'''
MSE Loss: the derivative would be getting really small considering that the values to be predicted are very small and so the mult factor should be very big
MAE Loss could be bigger than the MSE, this is a possible option.
'''
