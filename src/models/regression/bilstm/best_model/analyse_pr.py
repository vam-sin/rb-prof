# libraries
import pickle as pkl 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import torch 
from torch import nn
from model_utils import RBDataset, BiLSTMModel, train, evaluate, RBDataset_NoBS
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from scipy.stats import pearsonr
from os import listdir
from os.path import isfile, join
import sys
import logging
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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

def eval_pr(model: nn.Module, test_dataloader, filename, test_data, files_list) -> float:
    # print("Evaluating")
    model.eval()
    total_loss = 0 
    corr_lis = []
    len_lis = []
    with torch.no_grad():
        # print(i, tr_val[i])
        idx = files_list.index(filename)
        print(idx)
        inputs, labels = test_data.getitem(idx)
        inputs = inputs.float().to(device)
        # inputs = torch.unsqueeze(inputs, 2)
        # print(inputs.shape, labels.shape)

        labels = labels.float().to(device)
        labels = torch.squeeze(labels, 0)
        
        outputs = model(inputs)
        outputs = torch.squeeze(outputs, 0)
        outputs = torch.squeeze(outputs, 1)
        
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        y_pred_det = outputs.cpu().detach().numpy()
        y_true_det = labels.cpu().detach().numpy()

        corr_p, _ = pearsonr(y_true_det, y_pred_det) # y axis is predictions, x axis is true value
        for k in range(len(y_true_det)):
            len_lis.append(k)
            print(k, y_true_det[k], y_pred_det[k])
        
        logging.info(f'Corr: {corr_p:3.5f}')
        logging.info(f'Loss: {loss:3.5f}')
        y_pred_imp_seq_neg = [(-1*x)-0.001 for x in y_pred_det]
        y_true_imp_seq = [(x)+0.001 for x in y_true_det]
        
        df = pd.DataFrame({'Predicted':y_pred_imp_seq_neg,'True':y_true_imp_seq})
        g = sns.lineplot(data = df, palette = ['#F2AA4C', '#101820'], dashes=False)
        g.axhline(0.0)
        # sns.lineplot(x=len_lis, y=y_pred_imp_seq_neg, label='Predicted', color='#f0932b')
        # sns.lineplot(x=len_lis, y=y_true_imp_seq, label='True', color='#6ab04c')
        sns.despine(left=True, bottom=True)
        g.set(xticklabels=[], yticklabels=[])  # remove the tick labels
        g.tick_params(bottom=False, left=False)  # remove the ticks
        plt.legend()
        plt.show()
        plt.savefig("pr_figs/FULLDS_BiLSTM-1/test_0.png", format="png")



if __name__ == '__main__':
    # import data 
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    test_files = [mypath + '/' + f for f in onlyfiles]

    test_data = RBDataset(test_files)
    bs = 1
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=True) 
    print("Total Number of Test Samples: ", len(test_dataloader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    # Bi-LSTM
    input_dim = 804
    embedding_dim = 64
    hidden_dim = 256
    output_dim = 1
    n_layers = 4
    bidirectional = True
    dropout = 0.4
    model = BiLSTMModel(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs=1).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(pytorch_total_params)

    criterion = nn.L1Loss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)

    # Evaluation Metrics
    model.load_state_dict(torch.load('bl_bs1_noNorm_orig.pt'))
    model.eval()
    with torch.no_grad():
        print("------------- Validation -------------")
        eval_pr(model, test_dataloader, 'ENSMUST00000112172.3_LEU-ILE-VAL_.pkl', test_data, onlyfiles)


'''
MSE Loss: the derivative would be getting really small considering that the values to be predicted are very small and so the mult factor should be very big
MAE Loss could be bigger than the MSE, this is a possible option.
'''
