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
from perceiver_pytorch import PerceiverIO
from einops import rearrange, repeat

conds_dict = {'CTRL': 0, 'LEU': 1, 'ILE': 2, 'VAL': 3, 'LEU-ILE': 4, 'LEU-ILE-VAL': 5}

class PerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIO(
            dim = dim + 22,
            queries_dim = dim + 22,
            logits_dim = 1,
            **kwargs
        )

    def forward(
        self,
        x, conds_fin,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        # concatenate conds
        # print("x: ", x.shape)
        # print("conds_fin: ", conds_fin.shape)
        x = torch.cat((x, conds_fin), dim = 2)

        logits = self.perceiver_io(x, mask = mask, queries = x)
        return logits

class RBDataset_NoBS(Dataset):
    def __init__(self, list_of_file_paths, dataset_name, feature_list):
        self.list_of_file_paths = list_of_file_paths
        self.dataset_name = dataset_name
        self.feature_list = feature_list
        self.max_len = 10000

    def __len__(self):
        return len(self.list_of_file_paths)
    
    def __getitem__(self, index):
        with open(self.list_of_file_paths[index], 'rb') as f:
            data_dict = pkl.load(f)

        # get data
        X_seq = np.asarray(data_dict['sequence'])

        # # features
        if self.dataset_name == 'DS06':
            nt = np.asarray(data_dict['nt'])
            cbert = np.asarray(data_dict['cbert'])
            t5 = np.asarray(data_dict['t5'])
            lem = np.asarray(data_dict['lem'])
            mlm_cdna_nt = np.asarray(data_dict['mlm_cdna_nt_pbert'])
            mlm_cdna_nt_idai = np.asarray(data_dict['mlm_cdna_nt_idai'])
            af2 = np.asarray(data_dict['AF2-SS'])
            conds = np.asarray(data_dict['conds'])
            depr_vec = np.asarray(data_dict['depr_vec'])
            geom = np.asarray(data_dict['geom'])
            codon_epa_encodings = np.asarray(data_dict['codon_epa_encodings'])

            if 'AF2-SS' in self.feature_list or 't5' in self.feature_list or 'geom' in self.feature_list:
                len_ = af2.shape[0]
                nt = nt[:len_, :]
                cbert = cbert[:len_, :]
                lem = lem[:len_, :]
                mlm_cdna_nt = mlm_cdna_nt[:len_, :]
                mlm_cdna_nt_idai = mlm_cdna_nt_idai[:len_, :]
                conds = np.asarray(data_dict['conds'])[:len_, :]
                depr_vec = np.asarray(data_dict['depr_vec'])[:len_, :]

            if 'codon_epa_encodings' in self.feature_list:
                nt = nt[2:, :]
                conds = np.asarray(data_dict['conds'])[2:, :]
                depr_vec = np.asarray(data_dict['depr_vec'])[2:, :]

        elif self.dataset_name == 'DS04':
            X = np.asarray(data_dict['X'])
            nt = X[:,0:15]
            cbert = X[:,15:15+768]
            conds = X[:,15+768:15+768+20]
            depr_vec = X[:,15+768+20:15+768+20+1]

        conds_new = np.zeros((nt.shape[0], 21)) # additional to make it 21
        # put first 20 as prev conds
        conds_new[:,0:20] = conds
        if 'CTRL' in self.list_of_file_paths[index]:
            # set last one to 1
            conds_new[:,20] = 1

        # combine depr codons and conds
        conds_fin = np.concatenate((conds_new, depr_vec), axis=1)

        # combine features
        X_ft = conds_fin
        if 'AF2-SS' in self.feature_list:
            X_ft = np.concatenate((af2, X_ft), axis=1)
        if 'mlm_cdna_nt_idai' in self.feature_list:
            X_ft = np.concatenate((mlm_cdna_nt_idai, X_ft), axis=1)
        if 'mlm_cdna_nt_pbert' in self.feature_list:
            X_ft = np.concatenate((mlm_cdna_nt, X_ft), axis=1)
        if 'lem' in self.feature_list:
            X_ft = np.concatenate((lem, X_ft), axis=1)
        if 't5' in self.feature_list:
            X_ft = np.concatenate((t5, X_ft), axis=1)
        if 'cbert' in self.feature_list:
            X_ft = np.concatenate((cbert, X_ft), axis=1)
        if 'nt' in self.feature_list:
            X_ft = np.concatenate((nt, X_ft), axis=1)
        if 'geom' in self.feature_list:
            X_ft = np.concatenate((geom, X_ft), axis=1)
        if 'codon_epa_encodings' in self.feature_list:
            X_ft = np.concatenate((codon_epa_encodings, X_ft), axis=1)

        # y
        # y = np.absolute(data_dict['y'])[:len_]
        y = np.absolute(data_dict['y'])
        # y = y
        # min max norm
        # y = (y - np.min(y)) / (np.max(y) - np.min(y))

        if 'codon_epa_encodings' in self.feature_list:
            y = y[2:]

        if 'AF2-SS' in self.feature_list or 't5' in self.feature_list or 'geom' in self.feature_list:
            y = y[:len_]

        condition = self.list_of_file_paths[index].split('/')[-1].split('_')[1]

        if 'cembeds' in self.feature_list:
            return X_seq, conds_fin, y, condition
        
        return X_ft, conds_fin, y, condition

class PerceiverModel(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, num_latents, latent_dim, cross_heads, latent_heads, cross_dim_head, latent_dim_head, weight_tie_layers):
        super().__init__()

        self.model = PerceiverLM(
                        num_tokens = num_tokens,          # number of tokens
                        dim = dim,                    # dimension of sequence to be encoded
                        depth = depth,                   # depth of net
                        max_seq_len = max_seq_len,          # maximum sequence length
                        num_latents = num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
                        latent_dim = latent_dim,            # latent dimension
                        cross_heads = cross_heads,             # number of heads for cross attention. paper said 1
                        latent_heads = latent_heads,            # number of heads for latent self attention, 8
                        cross_dim_head = cross_dim_head,         # number of dimensions per cross attention head
                        latent_dim_head = latent_dim_head,        # number of dimensions per latent self attention head
                        weight_tie_layers = weight_tie_layers    # whether to weight tie layers (optional, as indicated in the diagram)
                    )
    
    def forward(self, src, conds_fin):
        x = self.model(src, conds_fin) # (batch, seq_len, dim)

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
        inputs = inputs.long().to(device)

        labels = labels.float().to(device)
        labels = torch.squeeze(labels, 0)

        conds_fin = conds_fin.float().to(device)
        
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

        if (i) % (100) == 0:
            logger.info(f'| samples trained: {(i+1)*bs:5d} | train (intermediate) loss: {total_loss/((i+1)*bs):5.10f} | train (intermediate) pr: {np.mean(pearson_corr_lis):5.10f} | train (intermediate) sr: {np.mean(spearman_corr_lis):5.10f} |')
    
    logger.info(f'Epoch Train Loss: {total_loss/len(train_dataloder): 5.10f} | train pr: {np.mean(pearson_corr_lis):5.10f} | train sr: {np.mean(spearman_corr_lis):5.10f} |')

    return total_loss/len(train_dataloder), np.mean(pearson_corr_lis), np.mean(spearman_corr_lis)

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
            # if perc_non_zero_labels >= 0.6:
            inputs = inputs.long().to(device)
            # embeds
            # inputs = inputs.long().to(device)
            # conds_fin = conds_fin.float().to(device)
            # condition_vec = torch.tensor(conds_dict[condition[0]]).long().to(device)

            labels = labels.float().to(device)
            labels = torch.squeeze(labels, 0)
            
            conds_fin = conds_fin.float().to(device)
        
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
    
    return total_loss / (len(val_dataloader) - 1), np.mean(corr_lis)
