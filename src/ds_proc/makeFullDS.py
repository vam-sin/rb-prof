'''
takes processed files and splits them into inidivudal transcript files with dictionaries containing the features.
'''

import numpy as np
import pickle as pkl
import os

three_to_one = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}

X_train = []
y_train = []

ctrl_condition_arr = np.zeros(20)

# codons: none

ile_condition_arr = np.zeros(20)
ile_condition_arr[9] = 1

# codons: ATT, ATC, ATA (3, 2, 1)

leu_condition_arr = np.zeros(20)
leu_condition_arr[10] = 1

# codons: TTA, TTG  (55, 56)

leu_ile_condition_arr = np.zeros(20)
leu_ile_condition_arr[9] = 1
leu_ile_condition_arr[10] = 1

# codons: TTA, TTG, ATT, ATC, ATA (55, 56, 3, 2, 1)

leu_ile_val_condition_arr = np.zeros(20)
leu_ile_val_condition_arr[9] = 1
leu_ile_val_condition_arr[10] = 1
leu_ile_val_condition_arr[19] = 1

# codons: TTA, TTG, ATT, ATC, ATA, GTT, GTC, GTA, GTG (55, 56, 3, 2, 1, 33, 34, 35, 36)

val_condition_arr = np.zeros(20)
val_condition_arr[19] = 1

# codons: GTT, GTC, GTA, GTG (33, 34, 35, 36)

output_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/train/'

# CTRL _ TRAIN

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_CTRL_train.pkl', 'rb') as f:
    ctrl_train = pkl.load(f)
train_ctrl_transcripts = list(ctrl_train.keys())

for x in train_ctrl_transcripts:
    ctrl_train[x].append(ctrl_condition_arr)

for x in train_ctrl_transcripts:
    condition_feat = []
    for i in range(len(ctrl_train[x][3])):
        condition_feat.append(ctrl_condition_arr)
    condition_feat = np.array(condition_feat)

    depr_codons = []
    for i in range(len(ctrl_train[x][0])):
        depr_codons.append(0)
    depr_codons = np.asarray(depr_codons)
    depr_codons = np.expand_dims(depr_codons, axis=1)
    # print(ctrl_train[x][3].shape, ctrl_train[x][4].shape, condition_feat.shape)
    X_sample = np.concatenate([ctrl_train[x][3], ctrl_train[x][4], condition_feat, depr_codons], axis=1)
    # X_train.append(X_sample)
    y_sample = np.asarray(ctrl_train[x][1])
    # y_train.append(y_sample)
    out_dict = {}
    out_dict['sequence'] = ctrl_train[x][0]
    out_dict['gene'] = ctrl_train[x][2]
    out_dict['X'] = X_sample
    out_dict['y'] = y_sample
    out_dict['condition'] = ctrl_condition_arr
    out_file_path = output_folder + x + '_CTRL_' + '.pkl'
    with open(out_file_path, 'wb') as f:
        pkl.dump(out_dict, f)
    # print(out_dict, out_file_path)
    # print(X_sample.shape, y_sample.shape)

print("CTRL Done")
# print("X_train: ", len(X_train), "y_train: ", len(y_train))

# ILE _ TRAIN

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_ILE_train.pkl', 'rb') as f:
    ile_train = pkl.load(f)
train_ile_transcripts = list(ile_train.keys())

for x in train_ile_transcripts:
    ile_train[x].append(ile_condition_arr)

for x in train_ile_transcripts:
    condition_feat = []
    for i in range(len(ile_train[x][3])):
        condition_feat.append(ile_condition_arr)
    condition_feat = np.array(condition_feat)

    depr_codons = []
    for i in range(len(ile_train[x][0])):
        if ile_train[x][0][i] in [1, 2, 3]:
            depr_codons.append(1)
        else:
            depr_codons.append(0)
    depr_codons = np.asarray(depr_codons)
    depr_codons = np.expand_dims(depr_codons, axis=1)
    # print(ile_train[x][3].shape, ile_train[x][4].shape, condition_feat.shape)
    X_sample = np.concatenate([ile_train[x][3], ile_train[x][4], condition_feat, depr_codons], axis=1)
    # X_train.append(X_sample)
    y_sample = np.asarray(ile_train[x][1])
    # y_train.append(y_sample)
    # print(X_sample.shape, y_sample.shape)
    out_dict = {}
    out_dict['sequence'] = ile_train[x][0]
    out_dict['gene'] = ile_train[x][2]
    out_dict['X'] = X_sample
    out_dict['y'] = y_sample
    out_dict['condition'] = ile_condition_arr
    out_file_path = output_folder + x + '_ILE_' + '.pkl'
    with open(out_file_path, 'wb') as f:
        pkl.dump(out_dict, f)

print("ILE Done")
# print("X_train: ", len(X_train), "y_train: ", len(y_train))

# LEU _ TRAIN

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_LEU_train.pkl', 'rb') as f:
    leu_train = pkl.load(f)
train_leu_transcripts = list(leu_train.keys())

for x in train_leu_transcripts:
    leu_train[x].append(leu_condition_arr)

for x in train_leu_transcripts:
    condition_feat = []
    for i in range(len(leu_train[x][3])):
        condition_feat.append(leu_condition_arr)
    condition_feat = np.array(condition_feat)

    depr_codons = []
    for i in range(len(leu_train[x][0])):
        if leu_train[x][0][i] in [55, 56]:
            depr_codons.append(1)
        else:
            depr_codons.append(0)
    depr_codons = np.asarray(depr_codons)
    depr_codons = np.expand_dims(depr_codons, axis=1)

    # print(leu_train[x][3].shape, leu_train[x][4].shape, condition_feat.shape)
    X_sample = np.concatenate([leu_train[x][3], leu_train[x][4], condition_feat, depr_codons], axis=1)
    # X_train.append(X_sample)
    y_sample = np.asarray(leu_train[x][1])
    # y_train.append(y_sample)
    # print(X_sample.shape, y_sample.shape)
    out_dict = {}
    out_dict['sequence'] = leu_train[x][0]
    out_dict['gene'] = leu_train[x][2]
    out_dict['X'] = X_sample
    out_dict['y'] = y_sample
    out_dict['condition'] = leu_condition_arr
    out_file_path = output_folder + x + '_LEU_' + '.pkl'
    with open(out_file_path, 'wb') as f:
        pkl.dump(out_dict, f)

print("LEU Done")
# print("X_train: ", len(X_train), "y_train: ", len(y_train))

# LEU + ILE _ TRAIN

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_LEU_ILE_train.pkl', 'rb') as f:
    leu_ile_train = pkl.load(f)
train_leu_ile_transcripts = list(leu_ile_train.keys())

for x in train_leu_ile_transcripts:
    leu_ile_train[x].append(leu_ile_condition_arr)

for x in train_leu_ile_transcripts:
    condition_feat = []
    for i in range(len(leu_ile_train[x][3])):
        condition_feat.append(leu_ile_condition_arr)
    condition_feat = np.array(condition_feat)

    depr_codons = []
    for i in range(len(leu_ile_train[x][0])):
        if leu_ile_train[x][0][i] in [3, 2, 1, 55, 56]:
            depr_codons.append(1)
        else:
            depr_codons.append(0)
    depr_codons = np.asarray(depr_codons)
    depr_codons = np.expand_dims(depr_codons, axis=1)

    # print(leu_ile_train[x][3].shape, leu_ile_train[x][4].shape, condition_feat.shape)
    X_sample = np.concatenate([leu_ile_train[x][3], leu_ile_train[x][4], condition_feat, depr_codons], axis=1)
    # X_train.append(X_sample)
    y_sample = np.asarray(leu_ile_train[x][1])
    # y_train.append(y_sample)
    # print(X_sample.shape, y_sample.shape)
    out_dict = {}
    out_dict['sequence'] = leu_ile_train[x][0]
    out_dict['gene'] = leu_ile_train[x][2]
    out_dict['X'] = X_sample
    out_dict['y'] = y_sample
    out_dict['condition'] = leu_ile_condition_arr
    out_file_path = output_folder + x + '_LEU-ILE_' + '.pkl'
    with open(out_file_path, 'wb') as f:
        pkl.dump(out_dict, f)

print("LEU-ILE Done")
# print("X_train: ", len(X_train), "y_train: ", len(y_train))

# LEU + ILE _VAL _ TRAIN

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_LEU_ILE_VAL_train.pkl', 'rb') as f:
    leu_ile_val_train = pkl.load(f)
train_leu_ile_val_transcripts = list(leu_ile_val_train.keys())

for x in train_leu_ile_val_transcripts:
    leu_ile_val_train[x].append(leu_ile_val_condition_arr)

for x in train_leu_ile_val_transcripts:
    condition_feat = []
    for i in range(len(leu_ile_val_train[x][3])):
        condition_feat.append(leu_ile_val_condition_arr)
    condition_feat = np.array(condition_feat)

    depr_codons = []
    for i in range(len(leu_ile_val_train[x][0])):
        if leu_ile_val_train[x][0][i] in [55, 56, 3, 2, 1, 33, 34, 35, 36]:
            depr_codons.append(1)
        else:
            depr_codons.append(0)
    depr_codons = np.asarray(depr_codons)
    depr_codons = np.expand_dims(depr_codons, axis=1)
    # print(leu_ile_val_train[x][3].shape, leu_ile_val_train[x][4].shape, condition_feat.shape)
    X_sample = np.concatenate([leu_ile_val_train[x][3], leu_ile_val_train[x][4], condition_feat, depr_codons], axis=1)
    # X_train.append(X_sample)
    y_sample = np.asarray(leu_ile_val_train[x][1])
    # y_train.append(y_sample)
    # print(X_sample.shape, y_sample.shape)
    out_dict = {}
    out_dict['sequence'] = leu_ile_val_train[x][0]
    out_dict['gene'] = leu_ile_val_train[x][2]
    out_dict['X'] = X_sample
    out_dict['y'] = y_sample
    out_dict['condition'] = leu_ile_val_condition_arr
    out_file_path = output_folder + x + '_LEU-ILE-VAL_' + '.pkl'
    with open(out_file_path, 'wb') as f:
        pkl.dump(out_dict, f)

print("LEU-ILE-VAL Done")
# print("X_train: ", len(X_train), "y_train: ", len(y_train))

# VAL _ TRAIN

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_VAL_train.pkl', 'rb') as f:
    val_train = pkl.load(f)
train_val_transcripts = list(val_train.keys())
# print(train_val_transcripts)
for x in train_val_transcripts:
    val_train[x].append(val_condition_arr)

for x in train_val_transcripts:
    condition_feat = []
    for i in range(len(val_train[x][3])):
        condition_feat.append(val_condition_arr)
    condition_feat = np.array(condition_feat)

    depr_codons = []
    for i in range(len(val_train[x][0])):
        if val_train[x][0][i] in [33, 34, 35, 36]:
            depr_codons.append(1)
        else:
            depr_codons.append(0)
    depr_codons = np.asarray(depr_codons)
    depr_codons = np.expand_dims(depr_codons, axis=1)
    # print(val_train[x][3].shape, val_train[x][4].shape, condition_feat.shape, depr_codons.shape)
    X_sample = np.concatenate([val_train[x][3], val_train[x][4], condition_feat, depr_codons], axis=1)
    # print(X_sample.shape)
    # X_train.append(X_sample)
    y_sample = np.asarray(val_train[x][1])
    # y_train.append(y_sample)
    # print(X_sample.shape, y_sample.shape)
    out_dict = {}
    out_dict['sequence'] = val_train[x][0]
    # print(val_train[x][0])
    out_dict['gene'] = val_train[x][2]
    out_dict['X'] = X_sample
    out_dict['y'] = y_sample
    out_dict['condition'] = val_condition_arr
    out_file_path = output_folder + x + '_VAL_' + '.pkl'
    with open(out_file_path, 'wb') as f:
        pkl.dump(out_dict, f)

print("VAL Done")
# print("X_train: ", len(X_train), "y_train: ", len(y_train))

# with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/X_full_val.pkl', 'wb') as f:
#     pkl.dump(X_train, f)

# with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/y_full_val.pkl', 'wb') as f:
#     pkl.dump(y_train, f)


'''
'''