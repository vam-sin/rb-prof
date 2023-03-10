# libraries
import pickle as pkl 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import torch 
from torch import nn
from models.regression.full_tf_model_utils import process_sample, TransformerModel
from scipy.stats import pearsonr
from os import listdir
from os.path import isfile, join
import sys
import logging
import matplotlib.pyplot as plt

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

def eval_pr(model: nn.Module, tr_val_key) -> float:
    # print("Evaluating")
    model.eval()
    total_loss = 0 
    corr_lis = []
    len_lis = []
    with torch.no_grad():
        # print(i, tr_val[i])
        X, y = process_sample(tr_val_key, mult_factor)
        # print(X.shape)
        seq_len = len(X)
        
        # src_mask = generate_square_subsequent_mask(seq_len).to(device)
        # print(src_mask.shape)
        x_in = torch.from_numpy(X).float().to(device)
        # src_mask = generate_square_subsequent_mask(seq_len).float().to(device)
        y_true = torch.from_numpy(np.expand_dims(y, axis=1)).float().to(device)
        y_pred = torch.squeeze(model(x_in, y_true), dim=1)
        y_true = torch.squeeze(y_true, dim=1)

        loss = criterion(y_pred, y_true)

        y_pred_imp_seq = []
        y_true_imp_seq = []
        for x in range(len(y_pred.cpu().detach().numpy())):
            # if y_true[x].item() != 0:
            y_pred_imp_seq.append(y_pred[x].item())
            y_true_imp_seq.append(y_true[x].item())

        corr, _ = pearsonr(y_true_imp_seq, y_pred_imp_seq) # y axis is predictions, x axis is true value
        for k in range(len(y_true_imp_seq)):
            len_lis.append(k)
            print(k, y_true_imp_seq[k], y_pred_imp_seq[k])
        
        logging.info(f'Corr: {corr:3.5f}')
        logging.info(f'Loss: {loss:3.5f}')
        y_pred_imp_seq_neg = [(-1*x)-0.001 for x in y_pred_imp_seq]
        y_true_imp_seq = [(x)+0.001 for x in y_true_imp_seq]
        
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
        plt.savefig("pr_figs/TF-Reg-Model-FULL_0/" + tr_val_key + ".png", format="png")

if __name__ == '__main__':
    # import data 
    with open('processed_keys/keys_proc_20c_60p.pkl', 'rb') as f:
        onlyfiles = pkl.load(f)
    print("Total Number of Samples: ", len(onlyfiles))

    print("---- Dataset Processing ----")
    tr_train, tr_test = train_test_split(onlyfiles, test_size=0.2, random_state=42, shuffle=True)
    tr_train, tr_val = train_test_split(tr_train, test_size=0.25, random_state=42, shuffle=True)

    print("Train Set: ", len(tr_train), "|| Validation Set: ", len(tr_val), "|| Test Set: " , len(tr_test))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    intoken = 1903 # input features
    outtoken = 1 # output 
    hidden = 256 # hidden layer size
    enc_layers = 3 # number of encoder layers
    dec_layers = 3 # number of decoder layers
    nhead = 8 # number of attention heads
    dropout = 0.1
    bs = 16 # batch_size
    model = TransformerModel(intoken, outtoken, nhead, hidden, enc_layers, dec_layers, dropout).to(device)
    model = model.to(torch.float)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(pytorch_total_params)

    criterion = nn.L1Loss()
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10, factor=0.1, verbose=True)

    # Evaluation Metrics
    model.load_state_dict(torch.load('reg_models/TF-Reg-Model-FULL_0.pt'))
    model.eval()
    with torch.no_grad():
        print("------------- Validation -------------")
        f = 4
        eval_pr(model, tr_test[10 * f])


'''
MSE Loss: the derivative would be getting really small considering that the values to be predicted are very small and so the mult factor should be very big
MAE Loss could be bigger than the MSE, this is a possible option.
'''
