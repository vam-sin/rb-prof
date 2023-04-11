# libraries
import pickle as pkl 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import torch 
from torch import nn
from full_ds_bilstm_model_utils import RBDataset_NoBS_withGene, BiLSTMModel
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

def evaluate(model: nn.Module, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger) -> float:
    print("Evaluating")
    model.eval()
    total_loss = 0. 
    corr_ctrl = []
    corr_ile = []
    corr_leu_ile = []
    corr_leu = []
    corr_leu_ile_val = []
    corr_val = []
    corr_lis = []
    perc_lis = []
    perc_window_lis = []
    perc_window_lis_2 = []
    tr_lis = []
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs, labels, tr, gene, f_name = data

            f_name = f_name[0]
            # print(f_name)
            tr = tr[0]
            gene = gene[0]
            # print(tr, gene)
            tr_lis.append(tr)
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

            # get percentage of reads
            non_zero = np.count_nonzero(y_true_det)
            perc = non_zero / len(y_true_det)
            # print(perc)
            perc_lis.append(perc)

            corr_p, _ = pearsonr(y_true_det, y_pred_det)
            
            corr_lis.append(corr_p)
            if perc >= 0.6 and perc < 0.8:
                perc_window_lis.append(corr_p)
            if perc >= 0.8:
                perc_window_lis_2.append(corr_p)

            if '_CTRL_' in f_name:
                corr_ctrl.append(corr_p)
            
            if '_ILE_' in f_name:
                corr_ile.append(corr_p)

            if '_LEU-ILE_' in f_name:
                corr_leu_ile.append(corr_p)

            if '_LEU_' in f_name:
                corr_leu.append(corr_p)

            if '_LEU-ILE-VAL_' in f_name:
                corr_leu_ile_val.append(corr_p)

            if '_VAL_' in f_name:
                corr_val.append(corr_p)

    logger.info(f'| PR: {np.mean(corr_lis):5.5f} |')
    # length of each of the six lists and the mean of the pearson correlation coefficient
    print("CTRL: ", len(corr_ctrl), np.mean(corr_ctrl))
    print("ILE: ", len(corr_ile), np.mean(corr_ile))
    print("LEU-ILE: ", len(corr_leu_ile), np.mean(corr_leu_ile))
    print("LEU: ", len(corr_leu), np.mean(corr_leu))
    print("LEU-ILE-VAL: ", len(corr_leu_ile_val), np.mean(corr_leu_ile_val))
    print("VAL: ", len(corr_val), np.mean(corr_val))
    print("Corr Perc Window 0.6 to 0.8: ", np.mean(perc_window_lis), len(perc_window_lis))
    print("Corr Perc Window 0.8+: ", np.mean(perc_window_lis_2), len(perc_window_lis_2))

    # histogram of the corr_lis 
    # plt.hist(corr_lis, bins=20)
    # plt.title("Histogram of Pearson Correlation Coefficients")
    # plt.xlabel("Pearson Correlation Coefficient")
    # plt.ylabel("Frequency")
    # plt.savefig("plots/BiLSTM-F-1-redo/pearson_corr_hist.png")

    # perc_lis vs corr_lis
    plt.scatter(perc_lis, corr_lis)
    plt.title("Percentage of Reads vs Pearson Correlation Coefficient")
    plt.xlabel("Percentage of Reads")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.savefig("plots/BiLSTM-F-1-redo/pearson_corr_perc.png")

    # histogram with six subplots
    # fig, axs = plt.subplots(2, 3, sharex=True, tight_layout=True)
    # axs[0, 0].hist(corr_ctrl, bins=20)
    # axs[0, 0].set_title('CTRL')
    # axs[0, 1].hist(corr_ile, bins=20)
    # axs[0, 1].set_title('ILE')
    # axs[0, 2].hist(corr_leu_ile, bins=20)
    # axs[0, 2].set_title('LEU-ILE')
    # axs[1, 0].hist(corr_leu, bins=20)
    # axs[1, 0].set_title('LEU')
    # axs[1, 1].hist(corr_leu_ile_val, bins=20)
    # axs[1, 1].set_title('LEU-ILE-VAL')
    # axs[1, 2].hist(corr_val, bins=20)
    # axs[1, 2].set_title('VAL')
    # for ax in axs.flat:
    #     ax.set(xlabel='P-Corr', ylabel='Frequency')
    #     ax.label_outer()
    # fig.tight_layout()

    # plt.savefig("plots/BiLSTM-F-1-redo/pearson_corr_hist_6ind.png")
    # plt.show()

    return total_loss / (len(val_dataloader) - 1)

if __name__ == '__main__':
    # import data 
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    test_files = [mypath + '/' + f for f in onlyfiles]

    test_data = RBDataset_NoBS_withGene(test_files)
    bs = 1
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=True) 
    print("Total Number of Test Samples: ", len(test_dataloader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    # Bi-LSTM
    input_dim = 803
    embedding_dim = 64
    hidden_dim = 256
    output_dim = 1
    n_layers = 4
    bidirectional = True
    dropout = 0.2
    bs = 1
    loss_mult_factor = 1
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
    model.load_state_dict(torch.load('reg_models/FULLDS_BiLSTM-Reg-1_redo.pt'))
    model.eval()
    with torch.no_grad():
        print("------------- Validation -------------")
        test_loss = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger)


'''BiLSTM-1
[Tue, 11 Apr 2023 16:25:51] INFO [model_plots.py.evaluate:116] | PR: 0.51291 |
CTRL:  336 0.4808924651288099
ILE:  175 0.5559916537848328
LEU-ILE:  92 0.5097523215573364
LEU:  163 0.5282413542944796
LEU-ILE-VAL:  8 0.6395002825886524
VAL:  0 nan

- results are lowest for CTRL and better for every other condition (which is excatly what we expected)

Corr Perc Window 0.6 to 0.8:  0.4932207979815279 544
Corr Perc Window 0.8+:  0.5594906529898218 230

- results are better with more %reads which is also expected. 
'''