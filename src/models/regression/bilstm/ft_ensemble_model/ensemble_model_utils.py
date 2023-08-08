# libraries
import numpy as np
from sklearn.model_selection import train_test_split
import torch 
from torch import nn
from scipy.stats import pearsonr, spearmanr 
from os import listdir
from os.path import isfile, join
import math
import pickle as pkl 
from torch.autograd import Variable
from torch.utils.data import Dataset

conds_dict = {'CTRL': 0, 'LEU': 1, 'ILE': 2, 'VAL': 3, 'LEU-ILE': 4, 'LEU-ILE-VAL': 5}

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class RBDataset_NoBS(Dataset):
    def __init__(self, list_of_file_paths, dataset_name, feature_list, models_dict):
        self.list_of_file_paths = list_of_file_paths
        self.dataset_name = dataset_name
        self.feature_list = feature_list
        self.models_dict = models_dict
        self.max_len = 10000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)

        # get data
        X_seq = np.asarray(data_dict['sequence'])

        # # features
        nt = np.asarray(data_dict['nt'])
        cbert = np.asarray(data_dict['cbert'])
        t5 = np.asarray(data_dict['t5'])
        lem = np.asarray(data_dict['lem'])
        mlm_cdna_nt_pbert = np.asarray(data_dict['mlm_cdna_nt_pbert'])
        mlm_cdna_nt_idai = np.asarray(data_dict['mlm_cdna_nt_idai'])
        af2 = np.asarray(data_dict['AF2-SS'])
        conds = np.asarray(data_dict['conds'])
        depr_vec = np.asarray(data_dict['depr_vec'])
        geom = np.asarray(data_dict['geom'])

        # truncate
        len_ = af2.shape[0]
        nt = nt[:len_, :]
        cbert = cbert[:len_, :]
        lem = lem[:len_, :]
        mlm_cdna_nt_pbert = mlm_cdna_nt_pbert[:len_, :]
        mlm_cdna_nt_idai = mlm_cdna_nt_idai[:len_, :]
        conds = np.asarray(data_dict['conds'])[:len_, :]
        depr_vec = np.asarray(data_dict['depr_vec'])[:len_, :]

        conds_new = np.zeros((nt.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        conds_new[:,0:20] = conds
        if 'CTRL' in self.list_of_file_paths[index]:
            # set last one to 1
            conds_new[:,20] = 1

        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_vec), axis=1)

        # get features
        nt_model_inp = np.concatenate((nt, conds_fin), axis=1)
        cbert_model_inp = np.concatenate((cbert, conds_fin), axis=1)
        t5_model_inp = np.concatenate((t5, conds_fin), axis=1)
        lem_model_inp = np.concatenate((lem, conds_fin), axis=1)
        mlm_cdna_nt_pbert_model_inp = np.concatenate((mlm_cdna_nt_pbert, conds_fin), axis=1)
        mlm_cdna_nt_idai_model_inp = np.concatenate((mlm_cdna_nt_idai, conds_fin), axis=1)
        af2_model_inp = np.concatenate((af2, conds_fin), axis=1)
        geom_model_inp = np.concatenate((geom, conds_fin), axis=1)

        # get preds from last layer of the models
        # nt
        self.models_dict['nt'].lin3.register_forward_hook(get_activation('lin3'))
        nt_preds = self.models_dict['nt'](torch.from_numpy(nt_model_inp).float().unsqueeze(0).cuda(), torch.from_numpy(conds_fin).float().unsqueeze(0).cuda())
        nt_preds = activation['lin3']
        # print(nt_preds.shape) # 1, len, 64

        # cbert
        self.models_dict['cbert'].lin3.register_forward_hook(get_activation('lin3'))
        cbert_preds = self.models_dict['cbert'](torch.from_numpy(cbert_model_inp).float().unsqueeze(0).cuda(), torch.from_numpy(conds_fin).float().unsqueeze(0).cuda())
        cbert_preds = activation['lin3']
        # print(cbert_preds.shape) # 1, len, 64

        # t5
        self.models_dict['t5'].lin3.register_forward_hook(get_activation('lin3'))
        t5_preds = self.models_dict['t5'](torch.from_numpy(t5_model_inp).float().unsqueeze(0).cuda(), torch.from_numpy(conds_fin).float().unsqueeze(0).cuda())
        t5_preds = activation['lin3']
        # print(t5_preds.shape) # 1, len, 64

        # lem
        self.models_dict['lem'].lin3.register_forward_hook(get_activation('lin3'))
        lem_preds = self.models_dict['lem'](torch.from_numpy(lem_model_inp).float().unsqueeze(0).cuda(), torch.from_numpy(conds_fin).float().unsqueeze(0).cuda())
        lem_preds = activation['lin3']
        # print(lem_preds.shape) # 1, len, 64

        # mlm_cdna_nt_pbert
        self.models_dict['mlm_cdna_nt_pbert'].lin3.register_forward_hook(get_activation('lin3'))
        mlm_cdna_nt_pbert_preds = self.models_dict['mlm_cdna_nt_pbert'](torch.from_numpy(mlm_cdna_nt_pbert_model_inp).float().unsqueeze(0).cuda(), torch.from_numpy(conds_fin).float().unsqueeze(0).cuda())
        mlm_cdna_nt_pbert_preds = activation['lin3']
        # print(mlm_cdna_nt_pbert_preds.shape) # 1, len, 64

        # mlm_cdna_nt_idai
        self.models_dict['mlm_cdna_nt_idai'].lin3.register_forward_hook(get_activation('lin3'))
        mlm_cdna_nt_idai_preds = self.models_dict['mlm_cdna_nt_idai'](torch.from_numpy(mlm_cdna_nt_idai_model_inp).float().unsqueeze(0).cuda(), torch.from_numpy(conds_fin).float().unsqueeze(0).cuda())
        mlm_cdna_nt_idai_preds = activation['lin3']
        # print(mlm_cdna_nt_idai_preds.shape) # 1, len, 64

        # af2
        self.models_dict['af2_ss'].lin3.register_forward_hook(get_activation('lin3'))
        af2_preds = self.models_dict['af2_ss'](torch.from_numpy(af2_model_inp).float().unsqueeze(0).cuda(), torch.from_numpy(conds_fin).float().unsqueeze(0).cuda())
        af2_preds = activation['lin3']
        # print(af2_preds.shape) # 1, len, 64

        # geom
        self.models_dict['geom'].lin3.register_forward_hook(get_activation('lin3'))
        geom_preds = self.models_dict['geom'](torch.from_numpy(geom_model_inp).float().unsqueeze(0).cuda(), torch.from_numpy(conds_fin).float().unsqueeze(0).cuda())
        geom_preds = activation['lin3']
        # print(geom_preds.shape) # 1, len, 64

        # concat all preds
        X_ft = torch.concatenate((nt_preds, cbert_preds, t5_preds, lem_preds, mlm_cdna_nt_pbert_preds, mlm_cdna_nt_idai_preds, af2_preds, geom_preds), axis=2)

        # squeeze X_ft
        X_ft = torch.squeeze(X_ft, axis=0)

        # y
        y = np.absolute(data_dict['y'])
        y = y[:len_] # truncate

        condition = self.list_of_file_paths[index].split('/')[-1].split('_')[1]
        
        return X_ft, conds_fin, y, condition

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, bs):
        super().__init__()

        # # embedding layer 
        # self.embedding = nn.Embedding(84, 256)

        # # context embedding
        # self.context_embedding = nn.Embedding(21, 783)

        # lstm model
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, dropout = dropout, batch_first=True)

        # params
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.bs = bs
        self.n_layers = n_layers

        # functions
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # linear layers
        self.lin1 = nn.Sequential(
            nn.Linear((hidden_dim * 2 if bidirectional else hidden_dim), 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lin2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lin3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.out_lin = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, src, condition):

        # # remove depr vector which is the last feature
        # depr_vec = src[:, :, -1]
        # depr_vec = torch.squeeze(depr_vec, axis=0)
        # src = src[:, :, :-1]
        # # print(src.shape)
        # # print(depr_vec.shape)

        # # context embedding
        # # get one the conds_fin
        # # print(condition)
        # # print(condition.shape)
        # conds_embed = self.context_embedding(condition)
        # # print(conds_embed.shape)

        # # iterate over depr_vec and if it is 1, add the conds embed to the src of that codon
        # for i in range(len(depr_vec)):
        #     if depr_vec[i] == 1:
        #         src[0][i] += conds_embed

        # embedding
        # src = self.embedding(src)
        # src = torch.cat((src, conds_fin), dim=2)

        # sequence analysis using LSTM
        h_0 = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(self.n_layers * 2 if self.bidirectional else self.n_layers, self.bs, self.hidden_dim).cuda()) # (1, bs, hidden)
        outputs, (final_hidden, final_cell) = self.lstm(src, (h_0, c_0))

        # append conds_vec to outputs
        # lin 1
        # x = torch.cat((outputs, conds_vec), dim=2)
        x = self.lin1(outputs)
        # x, _ = self.attn1(x, x, x)

        # lin 2
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.lin2(x)
        # x, _ = self.attn2(x, x, x)

        # lin 3
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.lin3(x)

        # out lin
        # x = torch.cat((x, conds_vec), dim=2)
        x = self.out_lin(x)

        return x

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()

        self.out_lin = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
    
    def forward(self, src, conds_fin):

        # out lin
        x = self.out_lin(src)

        return x


def train(model: nn.Module, train_dataloder, bs, device, criterion, mult_factor, loss_mult_factor, optimizer, logger) -> None:
    print("Training")
    model.train()
    total_loss = 0. 
    pearson_corr_lis = []
    spearman_corr_lis = []
    for i, data in enumerate(train_dataloder):
        optimizer.zero_grad()
        inputs, conds_fin, labels, condition = data
        # perc_sequence annotation
        len_ = inputs.shape[1]
        num_non_zero_labels = torch.sum(labels != 0.0)
        perc_non_zero_labels = num_non_zero_labels / (len_)
        # if perc_non_zero_labels < 0.6:
        inputs = inputs.float().to(device)
        conds_fin = conds_fin.float().to(device)
        # condition_vec = torch.tensor(conds_dict[condition[0]]).long().to(device)

        labels = labels.float().to(device)
        labels = torch.squeeze(labels, 0)
        
        outputs = model(inputs, conds_fin)
        outputs = torch.squeeze(outputs, 0)
        outputs = torch.squeeze(outputs, 1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * loss_mult_factor

        y_pred_det = outputs.cpu().detach().numpy()
        y_true_det = labels.cpu().detach().numpy()

        corr_p, _ = pearsonr(y_true_det, y_pred_det)
        corr_s, _ = spearmanr(y_true_det, y_pred_det)
        # print(corr_p)
        
        pearson_corr_lis.append(corr_p)
        spearman_corr_lis.append(corr_s)

        if (i) % (50) == 0:
            logger.info(f'| samples trained: {(i+1)*bs:5d} | train (intermediate) loss: {total_loss/((i+1)*bs):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')
    
    logger.info(f'Epoch Train Loss: {total_loss/len(train_dataloder): 5.10f} | train pr: {np.mean(pearson_corr_lis):5.10f} | train sr: {np.mean(spearman_corr_lis):5.10f} |')

def evaluate(model: nn.Module, val_dataloader, device, mult_factor, loss_mult_factor, criterion, logger, bs) -> float:
    print("Evaluating")
    model.eval()
    total_loss = 0. 
    corr_lis = []
    ctrl_corr_lis = []
    leu_corr_lis = []
    ile_corr_lis = []
    val_corr_lis = []
    leu_ile_corr_lis = []
    leu_ile_val_corr_lis = []
    file_preds = {}
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            inputs, conds_fin, labels, condition = data
            len_ = inputs.shape[1]
            num_non_zero_labels = torch.sum(labels != 0.0)
            perc_non_zero_labels = num_non_zero_labels / (len_)
            if perc_non_zero_labels >= 0.6:
                inputs = inputs.float().to(device)
                # embeds
                # inputs = inputs.long().to(device)
                conds_fin = conds_fin.float().to(device)
                # condition_vec = torch.tensor(conds_dict[condition[0]]).long().to(device)

                labels = labels.float().to(device)
                labels = torch.squeeze(labels, 0)
                
                outputs = model(inputs, conds_fin)
                outputs = torch.squeeze(outputs, 0)
                outputs = torch.squeeze(outputs, 1)

                loss = criterion(outputs, labels)

                total_loss += loss.item()

                y_pred_det = outputs.cpu().detach().numpy()
                y_true_det = labels.cpu().detach().numpy()

                corr_p, _ = pearsonr(y_true_det, y_pred_det)

                condition = condition[0]

                if condition == 'CTRL':
                    ctrl_corr_lis.append(corr_p)
                elif condition == 'LEU':
                    leu_corr_lis.append(corr_p)
                elif condition == 'ILE':
                    ile_corr_lis.append(corr_p)
                elif condition == 'VAL':
                    val_corr_lis.append(corr_p)
                elif condition == 'LEU-ILE':
                    leu_ile_corr_lis.append(corr_p)
                elif condition == 'LEU-ILE-VAL':
                    leu_ile_val_corr_lis.append(corr_p)
                
                corr_lis.append(corr_p)

                # # append preds and filename to dict 
                # file_preds[filename[0]] = y_pred_det

    logger.info(f'| Full PR: {np.mean(corr_lis):5.5f} |')
    logger.info(f'| CTRL PR: {np.mean(ctrl_corr_lis):5.5f} |')
    logger.info(f'| LEU PR: {np.mean(leu_corr_lis):5.5f} |')
    logger.info(f'| ILE PR: {np.mean(ile_corr_lis):5.5f} |')
    logger.info(f'| VAL PR: {np.mean(val_corr_lis):5.5f} |')
    logger.info(f'| LEU_ILE PR: {np.mean(leu_ile_corr_lis):5.5f} |')
    logger.info(f'| LEU_ILE_VAL PR: {np.mean(leu_ile_val_corr_lis):5.5f} |')

    # save preds
    # with open('file_preds.pkl', 'wb') as f:
    #     pkl.dump(file_preds, f)
    
    return total_loss / (len(val_dataloader) - 1)
