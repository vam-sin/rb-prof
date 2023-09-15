# libraries
import pickle as pkl 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import random
from torchmetrics import PearsonCorrCoef
import seaborn as sns
import torch 
from torch import nn
from model_utils import RBDataset_NoBS_withGene, BiLSTMModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer 
from scipy.stats import pearsonr
from os import listdir
from os.path import isfile, join
import sys
import logging
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

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

class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcc_l = PearsonCorrCoef().to(torch.device('cuda'))
        
    def forward(self, pred, actual):
        return -1 * self.pcc_l(pred, actual)

mult_factor = 1

def make_gene_plot(y_true_det, y_pred_det, filename, extra):
    # y_pred_imp_seq_neg = [(-1*x)-0.01 for x in y_pred_det]
    # y_true_imp_seq = [(x)+0.01 for x in y_true_det]
    y_true_det = y_true_det / 1e+6
    y_pred_det = y_pred_det / 1e+6
    
    df = pd.DataFrame({'Predicted':y_pred_det,'True':y_true_det})
    g = sns.lineplot(data = df, palette = ['#f1c40f', '#3498db'], dashes=False)
    # g.axhline(0.0)
    # sns.lineplot(x=len_lis, y=y_pred_imp_seq_neg, label='Predicted', color='#f0932b')
    # sns.lineplot(x=len_lis, y=y_true_imp_seq, label='True', color='#6ab04c')
    # sns.despine(left=True, bottom=True)
    # g.set(xticklabels=[], yticklabels=[])  # remove the tick labels
    # g.tick_params(bottom=False, left=False)  # remove the ticks
    plt.xlabel("Gene Sequence")
    plt.ylabel("Normalized Ribosome Counts")
    plt.legend()
    plt.show()
    plt.savefig("plots/" + str(extra) + '_' + str(filename) + ".png", format="png")

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
    max_pr = 0
    max_f_name = ''
    min_pr = 1e+10
    min_f_name = ''
    y_pred_det_best = []
    y_true_det_best = []
    y_pred_det_worst = []
    y_true_det_worst = []
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
            
            outputs = model(inputs, f_name)
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

            if corr_p > max_pr:
                max_pr = corr_p
                max_f_name = f_name
                y_pred_det_best = y_pred_det
                y_true_det_best = y_true_det
            
            if corr_p < min_pr:
                min_pr = corr_p
                min_f_name = f_name
                y_pred_det_worst = y_pred_det
                y_true_det_worst = y_true_det
            
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

    # min max scale
    y_pred_det_best = (y_pred_det_best - np.min(y_pred_det_best)) / (np.max(y_pred_det_best) - np.min(y_pred_det_best))
    y_true_det_best = (y_true_det_best - np.min(y_true_det_best)) / (np.max(y_true_det_best) - np.min(y_true_det_best))
    
    make_gene_plot(y_true_det_best, y_pred_det_best, max_f_name, extra='best')
    plt.clf()

    # min max scale
    y_pred_det_worst = (y_pred_det_worst - np.min(y_pred_det_worst)) / (np.max(y_pred_det_worst) - np.min(y_pred_det_worst))
    y_true_det_worst = (y_true_det_worst - np.min(y_true_det_worst)) / (np.max(y_true_det_worst) - np.min(y_true_det_worst))
    make_gene_plot(y_true_det_worst, y_pred_det_worst, min_f_name, extra='worst')
    plt.clf()

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

    # perc_lis vs corr_lis
    plt.scatter(perc_lis, corr_lis, color='#2ecc71')
    plt.title("Percentage of Reads vs Pearson Correlation Coefficient")
    plt.xlabel("Percentage of Reads")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.savefig("plots/pearson_corr_perc.png")
    plt.clf()

    # histogram of the corr_lis 
    plt.hist(corr_lis, bins=100, color='#2ecc71')
    plt.title(f'[All Conditions] Av PR: {np.mean(corr_lis):1.2f}', fontsize=20)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.savefig("plots/pearson_corr_hist.png")
    plt.clf()

    # histogram of the corr_lis ctrl
    plt.hist(corr_ctrl, bins=100, color='#f1c40f')
    plt.title(f'[CTRL] Av PR: {np.mean(corr_ctrl):1.2f}', fontsize=20)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.savefig("plots/pearson_corr_hist_ctrl.png")
    plt.clf()

    # histogram of the corr_lis ile
    plt.hist(corr_ile, bins=100, color='#e67e22')
    plt.title(f'[ILE] Av PR: {np.mean(corr_ile):1.2f}', fontsize=20)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.savefig("plots/pearson_corr_hist_ile.png")
    plt.clf()

    # histogram of the corr_lis leu-ile
    plt.hist(corr_leu_ile, bins=100, color='#3498db')
    plt.title(f'[LEU-ILE] Av PR: {np.mean(corr_leu_ile):1.2f}', fontsize=20)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.savefig("plots/pearson_corr_hist_leu_ile.png")
    plt.clf()

    # histogram of the corr_lis leu 
    plt.hist(corr_leu, bins=100, color='#2ecc71')
    plt.title(f'[LEU] Av PR: {np.mean(corr_leu):1.2f}', fontsize=20)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.savefig("plots/pearson_corr_hist_leu.png")
    plt.clf()

    # histogram of the corr_lis val
    plt.hist(corr_val, bins=100, color='#9b59b6')
    plt.title(f'[VAL] Av PR: {np.mean(corr_val):1.2f}', fontsize=20)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.savefig("plots/pearson_corr_hist_val.png")
    plt.clf()

    # histogram of the corr_lis leu-ile-val
    plt.hist(corr_leu_ile_val, bins=100, color='#e74c3c')
    plt.title(f'[LEU-ILE-VAL] Av PR: {np.mean(corr_leu_ile_val):1.2f}', fontsize=20)
    plt.xlabel("Pearson Correlation Coefficient", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.savefig("plots/pearson_corr_hist_leu-ile-val.png")
    plt.clf()

    # histogram with six subplots
    cond_ctrl = []
    for x in range(len(corr_ctrl)):
        cond_ctrl.append('CTRL')

    cond_ile = []
    for x in range(len(corr_ile)):
        cond_ile.append('ILE')
    
    cond_leu_ile = []
    for x in range(len(corr_leu_ile)):
        cond_leu_ile.append('LEU-ILE')

    cond_leu = []
    for x in range(len(corr_leu)):
        cond_leu.append('LEU')
    
    cond_val = []
    for x in range(len(corr_val)):
        cond_val.append('VAL')
    
    cond_leu_ile_val = []
    for x in range(len(corr_leu_ile_val)):
        cond_leu_ile_val.append('LEU-ILE-VAL')
    
    # conctaenate all the lists
    corr_all = corr_ctrl + corr_ile + corr_leu_ile + corr_leu + corr_val + corr_leu_ile_val
    cond_all = cond_ctrl + cond_ile + cond_leu_ile + cond_leu + cond_val + cond_leu_ile_val

    # create a dataframe
    df = pd.DataFrame(list(zip(corr_all, cond_all)), columns=['corr', 'cond'])

    # plot the histogram according to the condition using facetgrid
    g = sns.FacetGrid(df, col="cond", col_wrap=3, height=4, aspect=1)
    g.map(plt.hist, "corr", bins=100, color='#2ecc71')
    plt.savefig("plots/pearson_corr_hist_facetgrid.png")
    plt.clf()


    # fig, axs = plt.subplots(2, 3, sharex=True, tight_layout=True)
    # axs[0, 0].hist(corr_ctrl, bins=20, color='#f1c40f')
    # axs[0, 0].set_title(f'[CTRL] Av PR: {np.mean(corr_ctrl):1.2f}', fontsize=20)
    # axs[0, 1].hist(corr_ile, bins=20, color="#e67e22")
    # axs[0, 1].set_title(f'[ILE] Av PR: {np.mean(corr_ile):1.2f}', fontsize=20)
    # axs[0, 2].hist(corr_leu_ile, bins=20, color="#3498db")
    # axs[0, 2].set_title(f'[LEU-ILE] Av PR: {np.mean(corr_leu_ile):1.2f}', fontsize=20)
    # axs[1, 0].hist(corr_leu, bins=20, color="#2ecc71")
    # axs[1, 0].set_title(f'[LEU] Av PR: {np.mean(corr_leu):1.2f}', fontsize=20)
    # axs[1, 1].hist(corr_leu_ile_val, bins=20, color='#9b59b6')
    # axs[1, 1].set_title(f'[LEU-ILE-VAL] Av PR: {np.mean(corr_leu_ile_val):1.2f}', fontsize=20)
    # axs[1, 2].hist(corr_val, bins=20, color='#e74c3c')
    # axs[1, 2].set_title(f'[VAL] Av PR: {np.mean(corr_val):1.2f}', fontsize=20)
    # for ax in axs.flat:
    #     ax.set(xlabel='P-Corr', ylabel='Frequency')
    #     ax.label_outer()
    #     # set x limits
    #     ax.set_xlim([0.0, 1.0])
    #     # set y limits
    #     ax.set_ylim([0, 80])
    # # fig.tight_layout()

    # plt.savefig("plots/pearson_corr_hist_6ind.png")
    # plt.clf()
    # plt.show()

    # best and worst files
    print("Best File: ", max_f_name, max_pr)
    print("Worst File: ", min_f_name, min_pr)

    return total_loss / (len(val_dataloader) - 1)

if __name__ == '__main__':
    # import data 
    mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    test_files_full = [mypath + '/' + f for f in onlyfiles]

    # test_files = []
    # for f in test_files_full:
    #     if '_ILE_' in f:
    #         test_files.append(f)

    test_data = RBDataset_NoBS_withGene(test_files_full)
    bs = 1
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=True) 
    print("Total Number of Test Samples: ", len(test_dataloader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    # force_cudnn_initialization()

    # Bi-LSTM
    input_dim = 805
    hidden_dim = 256
    output_dim = 1
    n_layers = 4
    bidirectional = True
    dropout = 0.1
    loss_mult_factor = 1
    model = BiLSTMModel(input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(pytorch_total_params)

    criterion = PCCLoss().to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)

    # Evaluation Metrics
    model.load_state_dict(torch.load('reg_models/B-0-DS06-NT_CBERT-BS1-PCCLoss.pt'))
    model.eval()
    with torch.no_grad():
        print("------------- Validation -------------")
        test_loss = evaluate(model, test_dataloader, device, mult_factor, loss_mult_factor, criterion, logger)


'''BiLSTM:
CTRL:  402 0.47280696905413766
ILE:  245 0.5190610245553468
LEU-ILE:  146 0.4468298206548672
LEU:  232 0.4817325897046564
LEU-ILE-VAL:  10 0.5313332937712453
VAL:  8 0.4967031421233196
Corr Perc Window 0.6 to 0.8:  0.4691532967579137 694
Corr Perc Window 0.8+:  0.5098339178441865 349
Best File:  ENSMUST00000020118.4_CTRL_.pkl 0.8851913939504836
Worst File:  ENSMUST00000084125.9_ILE_.pkl 0.09042225656963047
'''