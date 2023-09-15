# libraries
import os
import pandas as pd 
import numpy as np
import torch
import math
import torch.nn as nn
from torch.utils.data import Dataset
import csv
from transformers import Trainer
from torchmetrics import PearsonCorrCoef
import itertools
import pickle as pkl
from torch.autograd import Variable
import h5py
import pywt
from sklearn.model_selection import train_test_split

def RiboDatasetGWS(ribo_data_dirpath: str, ctrl_depr_path: str, ds: str, threshold: float = 0.6):
    if ds == 'Liver':
        # liver data
        df_liver = pd.read_csv(ribo_data_dirpath)

        df_liver.columns = ['gene', 'sequence', 'annotations', 'perc_non_zero_annots']
        # apply annot threshold 
        df_liver = df_liver[df_liver['perc_non_zero_annots'] >= threshold]

        df_train, df_test = train_test_split(df_liver, test_size=0.2, random_state=42)

        df_train.to_csv('data/ribo_train_liver.csv', index=False)
        df_test.to_csv('data/ribo_test_liver.csv', index=False)

        return df_train, df_test


    elif ds == 'DeprCTRL':
        # ctrl depr data
        df_ctrl_depr = pd.read_csv(ctrl_depr_path)
        # drop transcript column
        df_ctrl_depr = df_ctrl_depr.drop(columns=['transcript'])
        df_ctrl_depr.columns = ['gene', 'transcript', 'sequence', 'annotations', 'perc_non_zero_annots']
        # apply annot threshold
        df_full = df_ctrl_depr[df_ctrl_depr['perc_non_zero_annots'] >= threshold]

        genes = df_full['gene'].unique()
        genes_train, genes_test = train_test_split(genes, test_size=0.2, random_state=42)

        # split the dataframe
        df_train = df_full[df_full['gene'].isin(genes_train)]
        df_test = df_full[df_full['gene'].isin(genes_test)]

        df_train.to_csv('data/ribo_train_deprctrl.csv', index=False)
        df_test.to_csv('data/ribo_test_deprctrl.csv', index=False)

        return df_train, df_test

    elif ds == 'Liver_DeprCTRL':
        df_liver = pd.read_csv(ribo_data_dirpath)
        df_ctrl_depr = pd.read_csv(ctrl_depr_path)

        # add to the liver data the genes from ctrl depr which are not in liver
        genes_liver = df_liver['gene'].unique()
        genes_ctrl_depr = df_ctrl_depr['gene'].unique()
        genes_to_add = [gene for gene in genes_ctrl_depr if gene not in genes_liver]

        df_ctrl_depr = df_ctrl_depr[df_ctrl_depr['gene'].isin(genes_to_add)]

        df_full = pd.concat([df_liver, df_ctrl_depr], axis=0)

        df_full.columns = ['gene', 'sequence', 'annotations', 'perc_non_zero_annots', 'transcript']
        # apply annot threshold
        df_full = df_full[df_full['perc_non_zero_annots'] >= threshold]

        genes = df_full['gene'].unique()
        genes_train, genes_test = train_test_split(genes, test_size=0.2, random_state=42)

        # split the dataframe
        df_train = df_full[df_full['gene'].isin(genes_train)]
        df_test = df_full[df_full['gene'].isin(genes_test)]

        df_train.to_csv('data/ribo_train_liver_deprctrl.csv', index=False)
        df_test.to_csv('data/ribo_test_liver_deprctrl.csv', index=False)

        return df_train, df_test

class GWSDatasetFromPandas(Dataset):
    def __init__(self, df):
        self.df = df
        self.counts = list(self.df['annotations'])
        self.sequences = list(self.df['sequence'])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        X = self.df['sequence'].iloc[idx]
        # convert to int
        X = X[1:-1].split(', ')
        X = [int(i) for i in X]

        y = self.df['annotations'].iloc[idx]
        # convert string into list of floats
        y = y[1:-1].split(', ')
        y = [float(i) for i in y]

        # do min max scaling
        # y = [(i - min(y)) / (max(y) - min(y)) for i in y] # DO NOT MIN MAX SCALE
        # log transform y
        # add 1 to avoid log(0)
        y = [1+i for i in y]
        y = np.log(y)

        # pywt transform
        # get nan ids
        nan_ids = np.argwhere(np.isnan(y))

        # set nans to 0
        y = np.nan_to_num(y)
        
        # get approx and details from pywt haar
        wavelet = 'haar'
        cA, cD = pywt.dwt(y, wavelet)
        only_approx = pywt.idwt(cA, None, wavelet)
        only_details = pywt.idwt(None, cD, wavelet)
        only_approx = np.asarray(only_approx)
        only_details = np.asarray(only_details)

        # get x, y, only_approx, only_details in the same length (truncate to get the same length)
        # smallest length
        min_len = min(len(X), len(y), len(only_approx), len(only_details))
        X = X[:min_len]
        y = y[:min_len]
        only_approx = only_approx[:min_len]
        only_details = only_details[:min_len]

        idtw_y = only_approx + only_details

        # for x in range(len(idtw_y)):
        #     print(y[x], idtw_y[x])

        assert np.allclose(y, idtw_y, rtol=1e-05, atol=1e-08, equal_nan=True)

        # set nans back to nans
        only_approx[nan_ids] = np.nan
        only_details[nan_ids] = np.nan
        y[nan_ids] = np.nan
        idtw_y[nan_ids] = np.nan

        # # set the first and last 5 codons to na (setting both is not working)
        # y[:5] = [np.nan] * 5
        # only_approx[:5] = [np.nan] * 5
        # only_details[:5] = [np.nan] * 5

        # y[-5:] = [np.nan] * 5

        X = np.array(X)
        y = np.array(y)
        only_approx = np.array(only_approx)
        only_details = np.array(only_details)
        idtw_y = np.array(idtw_y)

        # # remove first five and last 5 codons (not focusing on those codons)
        # X = X[5:-5]
        # y = y[5:-5]

        X = torch.from_numpy(X).long()
        y = torch.from_numpy(y).float()
        only_approx = torch.from_numpy(only_approx).float()
        only_details = torch.from_numpy(only_details).float()
        idtw_y = torch.from_numpy(idtw_y).float()

        return X, y, only_approx, only_details, idtw_y

class MaskedPearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)
        return 1 - cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        # print(y_pred_mask, y_true_mask)

        loss = nn.functional.mse_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())

class MaskedPoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        return nn.functional.poisson_nll_loss(y_pred_mask, y_true_mask, log_input=False)

class MaskedCombinedPearsonLoss(nn.Module):
    def __init__(self, comb_max_duration: int = 200):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.poisson = MaskedPoissonLoss()
        self.comb_max_duration = comb_max_duration

    def __call__(self, y_pred, y_true, mask, timestamp, eps=1e-6):
        poisson = self.poisson(y_pred, y_true, mask)
        pearson = self.pearson(y_pred, y_true, mask, eps=eps)

        return pearson + max(0, 1 - timestamp / self.comb_max_duration) * poisson

class MaskedCombinedPearsonMSELossWavelet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pearson = MaskedPearsonLoss()
        self.l1 = MaskedL1Loss()

    def __call__(self, y_pred, y_true_approx, y_true_details, idwt_y, y_true_full,  mask, eps=1e-6):
        y_pred = torch.squeeze(y_pred, 0)
        approx_pred = y_pred[:, 0]
        details_pred = y_pred[:, 1]
        full_pred = approx_pred + details_pred

        l1_details = self.l1(details_pred, y_true_details, mask)
        pearson_approx = self.pearson(approx_pred, y_true_approx, mask, eps=eps)

        pearson_full = self.pearson(full_pred, idwt_y, mask, eps=eps)

        # print(mse_details, pearson_approx, pearson_full)

        return l1_details + pearson_approx + pearson_full

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs)
        labels = inputs.pop("labels") # y true full
        approx_labels = inputs.pop("approx") # y true approx
        details_labels = inputs.pop("details") # y true details
        idwt_y = inputs.pop("idwt_y") # y true idwt
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.squeeze(logits, dim=2)
        lengths = inputs['lengths']

        # loss_fnc = MaskedCombinedPearsonLoss()

        # loss_fnc = MaskedPearsonLoss()

        loss_fnc = MaskedCombinedPearsonMSELossWavelet()
        
        mask = torch.arange(logits.shape[1])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(labels)))
        
        # loss = loss_fnc(logits, labels, mask, self.state.epoch)
        
        # loss = loss_fnc(logits, labels, mask)

        # pywt loss
        # print("logits", logits)
        loss = loss_fnc(logits, approx_labels, details_labels, idwt_y, labels, mask)

        return (loss, outputs) if return_outputs else loss 