import numpy as np
import matplotlib.pyplot as plt
# import pearson
from scipy.stats import pearsonr, spearmanr 
import pandas as pd
import torch
from torch import nn
import os 
import sys
from torchmetrics.functional import pearson_corrcoef
import itertools

condition_values = {'CTRL': 0, 'ILE': 1, 'LEU': 2, 'VAL': 3, 'LEU_ILE': 4, 'LEU_ILE_VAL': 5}
inverse_condition_values = {0: 'CTRL', 1: 'ILE', 2: 'LEU', 3: 'VAL', 4: 'LEU_ILE', 5: 'LEU_ILE_VAL'}

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def pearson_mask(pred, label, mask, length):
    full_pred_tensor = torch.tensor(pred)
    label_tensor = torch.tensor(label)
    # make mask from the nans in label tensor
    mask = label_tensor != -100.0
    mask = torch.tensor(mask)
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(label_tensor)))

    # set full pred to same length as label tensor
    full_pred_tensor = full_pred_tensor[:len(mask)]

    # print(full_pred_tensor.shape, label_tensor.shape, mask.shape)

    assert full_pred_tensor.shape == label_tensor.shape
    assert label_tensor.shape == mask.shape

    mp, mt = torch.masked_select(full_pred_tensor, mask), torch.masked_select(label_tensor, mask)
    temp_pearson = pearson_corrcoef(mp, mt)

    # get float value from tensor
    temp_pearson = temp_pearson.item()

    # print(mp, mt)
    # print("Pearson Correlation Coefficient: ", temp_pearson)

    return temp_pearson


def analyse_dh_outputs(preds, labels, inputs, output_loc):
    # make mask removing those that have a input of -100
    mask = inputs != -100.0

    # get lengths of each sequence
    lengths = np.sum(mask, axis=1)

    # convert to lists and remove padding
    preds = preds.tolist()
    labels = labels.tolist()
    inputs = inputs.tolist()

    preds = [pred[:lengths[i]] for i, pred in enumerate(preds)]
    labels = [label[:lengths[i]] for i, label in enumerate(labels)]
    inputs = [input[:lengths[i]] for i, input in enumerate(inputs)]

    condition_samples = []

    # get conditions for each sample
    # do a / with 64 to get the condition
    for i in range(len(inputs)):
        condition_samples.append(inputs[i][0] // 64)

    labels_ctrl = []
    genes = []
    transcripts = []

    ds = pd.read_csv('data/ribo_test_ALL_dh_0.3_NZ_20_PercNan_0.05.csv')

    sequences_ds = []

    sequence_list = list(ds['sequence'])
    ctrl_sequence_list_ds = list(ds['ctrl_sequence'])
    genes_list_ds = list(ds['gene'])
    transcripts_list_ds = list(ds['transcript'])
    condition_list = list(ds['condition'])
    codon_sequences = []

    for i in range(len(sequence_list)):
        x = sequence_list[i][1:-1].split(', ')
        x = [int(i) for i in x]
        cond_val = condition_values[condition_list[i]]
        # get codon sequence from x
        codon_seq = [id_to_codon[i] for i in x]
        # convert to string
        codon_seq = ''.join(codon_seq)
        codon_sequences.append(codon_seq)
        # get the remainder
        add_val = (cond_val) * 64
        x = [i + add_val for i in x]
        sequences_ds.append(x)

    # get ctrl labels for each sample using the inputs
    
    # print(len(inputs), len(condition_samples), len(preds), len(labels))
    # print(inputs[1])

    for i in range(len(inputs)):
        condition_sample = inverse_condition_values[condition_samples[i]]
        # search for inputs[i] in sequences_ds get index
        for j in range(len(sequences_ds)):
            if sequences_ds[j] == inputs[i] and condition_sample == condition_list[j]:
                index = j
                break

        # if condition_sample == 'CTRL':
        #     labels_ctrl.append(labels[i])
        # else:
        # get ctrl label
        ctrl_sample = ctrl_sequence_list_ds[index]
        ctrl_sample = ctrl_sample[1:-1].split(', ')
        ctrl_sample = [float(k) for k in ctrl_sample]
        labels_ctrl.append(ctrl_sample)
        genes.append(genes_list_ds[index])
        transcripts.append(transcripts_list_ds[index])

    # labels ctrl mask them with length
    # labels_ctrl = []
    # for i in range(len(labels_ctrl_full)):
    #     labels_ctrl.append(labels_ctrl_full[i][:lengths[i]])

    # first dim of pred is ctrl, second is depr difference

    # ctrl predictions
    ctrl_preds = []
    for i in range(len(preds)):
        pred_sample = preds[i]
        pred_sample = np.asarray(pred_sample)
        # get first dim
        pred_sample = pred_sample[:, 0]
        ctrl_preds.append(pred_sample)

    depr_diffs = []
    for i in range(len(preds)):
        pred_sample = preds[i]
        pred_sample = np.asarray(pred_sample)
        # get second dim
        pred_sample = pred_sample[:, 1]
        depr_diffs.append(pred_sample)

    full_preds = []
    for i in range(len(preds)):
        full_preds.append(ctrl_preds[i] + depr_diffs[i])
        # print(len(full_preds[i]), len(labels[i]))

    # np log 1 + x the labels
    # labels = [np.log1p(label) for label in labels]
    labels_ctrl = [np.log1p(label) for label in labels_ctrl]

    # plot ten best samples
    # get pearson corr for each sample
    pearson_corrs = []
    for i in range(len(full_preds)):
        pearson_corrs.append(pearson_mask(full_preds[i], labels[i], mask[i], lengths[i]))

    pearson_corrs_ctrl = []
    for i in range(len(ctrl_preds)):
        pearson_corrs_ctrl.append(pearson_mask(ctrl_preds[i], labels_ctrl[i], mask[i], lengths[i]))

    # output all the predictions into df from lists
    output_analysis_df = pd.DataFrame(list(zip(transcripts, genes, codon_sequences, pearson_corrs, pearson_corrs_ctrl, condition_list)), columns =['Transcript', 'Gene', 'Sequence', 'Full Prediction Pearson Correlation', 'Control Prediction Pearson Correlation', 'Deprivation Condition'])
    output_analysis_df.to_csv(output_loc + "/analysis.csv", index=False)

    # get ten best samples
    best_samples = sorted(range(len(pearson_corrs)), key = lambda sub: pearson_corrs[sub])[-10:]
    # print best pearson corrs
    print("Best Pearson Correlations: ", [pearson_corrs[i] for i in best_samples])


    for i in range(10):
        out_loc = output_loc + "/full_plots/sample_" + str(best_samples[i]) + '_' + str(inverse_condition_values[condition_samples[best_samples[i]]]) + "_best_" + transcripts[best_samples[i]] + "_" + genes[best_samples[i]] + ".png"
        # pearson_corr_full, _ = pearsonr(full_preds[best_samples[i]], labels[best_samples[i]])
        # pearson_corr_ctrl, _ = pearsonr(ctrl_preds[best_samples[i]], labels[best_samples[i]])
        pearson_corr_full = pearson_mask(full_preds[best_samples[i]], labels[best_samples[i]], mask[best_samples[i]], lengths[best_samples[i]])
        pearson_corr_ctrl = pearson_mask(ctrl_preds[best_samples[i]], labels_ctrl[best_samples[i]], mask[best_samples[i]], lengths[best_samples[i]])

        print("Transcript: ", transcripts[best_samples[i]], " Gene: ", genes[best_samples[i]], " Condition: ", inverse_condition_values[condition_samples[best_samples[i]]], " Pearson Correlation FULL: ", pearson_corr_full, " Pearson Correlation CTRL: ", pearson_corr_ctrl)
        # print shapes
        print("Full Preds: ", np.asarray(full_preds[best_samples[i]]).shape, " CTRL Preds: ", np.asarray(ctrl_preds[best_samples[i]]).shape, " Labels CTRL: ", np.asarray(labels_ctrl[best_samples[i]]).shape, "Labels Full: ", np.asarray(labels[best_samples[i]]).shape, " Mask: ", np.asarray(mask[best_samples[i]]).shape, " Lengths: ", lengths[best_samples[i]])
        print("Labels CTRL: ", labels_ctrl[best_samples[i]], "Labels Full: ", labels[best_samples[i]])
        min_y = min(min(full_preds[best_samples[i]]), min(depr_diffs[best_samples[i]]), min(ctrl_preds[best_samples[i]])) - 0.1
        max_y = max(max(full_preds[best_samples[i]]), max(depr_diffs[best_samples[i]]), max(ctrl_preds[best_samples[i]])) + 0.1
        # subplots for ctrl, depr, full, labels
        fig, axs = plt.subplots(6, 1, figsize=(20, 10))
        axs[0].set_title("Pearson Correlation FULL: " + str(pearson_corr_full) + " Pearson Correlation CTRL: " + str(pearson_corr_ctrl) + " Condition: " + str(inverse_condition_values[condition_samples[best_samples[i]]]))
        # remove axes for axs[0]
        axs[0].axis('off')
        axs[1].plot(ctrl_preds[best_samples[i]], color='#2ecc71')
        # axs[1].set_title("CTRL PRED")
        # set limit to the max and min of full preds
        axs[1].set_ylim([min_y, max_y])

        axs[2].plot(depr_diffs[best_samples[i]], color='#e74c3c')
        # axs[2].set_title("DEPR DIFF PRED")
        axs[2].set_ylim([min_y, max_y])

        axs[3].plot(full_preds[best_samples[i]], color='#3498db')
        axs[3].set_ylim([min_y, max_y])
        # axs[3].set_title("FULL PRED")

        axs[4].plot(labels[best_samples[i]], color='#f39c12')
        # axs[4].set_title("LABEL FULL")

        axs[5].plot(labels_ctrl[best_samples[i]], color='#f39c12')
        # axs[5].set_title("LABEL CTRL")

        fig.tight_layout()

        plt.savefig(out_loc)
        plt.clf()

        # # make a folder for the transcript
        # cmd = "mkdir " + output_loc + "/" + str(transcripts[best_samples[i]])
        # os.system(cmd)

        # # make individual plots for each transcript for full preds, ctrl preds, depr diff preds, labels
        # # plot ctrl preds
        # out_loc = output_loc + "/" + str(transcripts[best_samples[i]]) + "/ctrl_preds.png"
        # plt.plot(ctrl_preds[best_samples[i]], color='#2ecc71')
        # plt.title("Control Prediction", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # # set y lims
        # plt.ylim([min_y, max_y])
        # plt.savefig(out_loc)
        # plt.clf()

        # # plot depr diff preds
        # out_loc = output_loc + "/" + str(transcripts[best_samples[i]]) + "/depr_diff_preds.png"
        # plt.plot(depr_diffs[best_samples[i]], color='#e74c3c')
        # plt.title("Deprivation Difference Prediction", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.ylim([min_y, max_y])
        # plt.savefig(out_loc)
        # plt.clf()

        # # plot full preds
        # out_loc = output_loc + "/" + str(transcripts[best_samples[i]]) + "/full_preds.png"
        # plt.plot(full_preds[best_samples[i]], color='#3498db')
        # plt.title("Full Prediction", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.ylim([min_y, max_y])
        # plt.savefig(out_loc)
        # plt.clf()

        # # plot labels
        # out_loc = output_loc + "/" + str(transcripts[best_samples[i]]) + "/labels_full.png"
        # plt.plot(labels[best_samples[i]], color='#f39c12')
        # plt.title("Full Labels", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.savefig(out_loc)
        # plt.clf()

        # # plot ctrl labels
        # out_loc = output_loc + "/" + str(transcripts[best_samples[i]]) + "/labels_ctrl.png"
        # plt.plot(labels_ctrl[best_samples[i]], color='#f39c12')
        # plt.title("Control Labels", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.savefig(out_loc)
        # plt.clf()


    # plot ten worst samples
    worst_samples = sorted(range(len(pearson_corrs)), key = lambda sub: pearson_corrs[sub])[:10]
    # print worst pearson corrs
    print("Worst Pearson Correlations: ", [pearson_corrs[i] for i in worst_samples])
    # get idx of the worst samples 

    for i in range(10):
        out_loc = output_loc + "/full_plots/sample_" + str(worst_samples[i]) + '_' + str(inverse_condition_values[condition_samples[worst_samples[i]]]) + "_worst_" + transcripts[worst_samples[i]] + "_" + genes[worst_samples[i]] + ".png"
        # pearson_corr_full, _ = pearsonr(full_preds[worst_samples[i]], labels[worst_samples[i]])
        # pearson_corr_ctrl, _ = pearsonr(ctrl_preds[worst_samples[i]], labels[worst_samples[i]])
        pearson_corr_full = pearson_mask(full_preds[worst_samples[i]], labels[worst_samples[i]], mask[worst_samples[i]], lengths[worst_samples[i]])
        pearson_corr_ctrl = pearson_mask(ctrl_preds[worst_samples[i]], labels_ctrl[worst_samples[i]], mask[worst_samples[i]], lengths[worst_samples[i]])
        min_y = min(min(full_preds[worst_samples[i]]), min(depr_diffs[worst_samples[i]]), min(ctrl_preds[worst_samples[i]])) - 0.1
        max_y = max(max(full_preds[worst_samples[i]]), max(depr_diffs[worst_samples[i]]), max(ctrl_preds[worst_samples[i]])) + 0.1

        print("Transcript: ", transcripts[worst_samples[i]], " Gene: ", genes[worst_samples[i]], " Condition: ", inverse_condition_values[condition_samples[worst_samples[i]]], " Pearson Correlation FULL: ", pearson_corr_full, " Pearson Correlation CTRL: ", pearson_corr_ctrl)
        # print shapes
        print("Full Preds: ", np.asarray(full_preds[best_samples[i]]).shape, " CTRL Preds: ", np.asarray(ctrl_preds[worst_samples[i]]).shape, " Labels CTRL: ", np.asarray(labels_ctrl[worst_samples[i]]).shape, "Labels Full: ", np.asarray(labels[worst_samples[i]]).shape, " Mask: ", np.asarray(mask[worst_samples[i]]).shape, " Lengths: ", lengths[worst_samples[i]])

        # subplots for ctrl, depr, full, labels
        fig, axs = plt.subplots(6, 1, figsize=(20, 10))
        axs[0].set_title("Pearson Correlation FULL: " + str(pearson_corr_full) + " Pearson Correlation CTRL: " + str(pearson_corr_ctrl) + " Condition: " + str(inverse_condition_values[condition_samples[worst_samples[i]]]))
        # remove axes for axs[0]
        axs[0].axis('off')
        axs[1].plot(ctrl_preds[worst_samples[i]], color='#2ecc71')
        # axs[1].set_title("CTRL")
        # set y lims to full preds
        axs[1].set_ylim([min_y, max_y])

        axs[2].plot(depr_diffs[worst_samples[i]], color='#e74c3c')
        # axs[2].set_title("DEPR DIFF")
        axs[2].set_ylim([min_y, max_y])

        axs[3].plot(full_preds[worst_samples[i]], color='#3498db')
        # axs[3].set_title("FULL PRED")
        axs[3].set_ylim([min_y, max_y])

        axs[4].plot(labels[worst_samples[i]], color='#f39c12')
        # axs[4].set_title("LABEL FULL")

        axs[5].plot(labels_ctrl[worst_samples[i]], color='#f39c12')
        # axs[5].set_title("LABEL CTRL")

        fig.tight_layout()

        # set title to pearson corr and condition

        plt.savefig(out_loc)
        plt.clf()

        # # make a folder for the transcript
        # cmd = "mkdir " + output_loc + "/" + str(transcripts[worst_samples[i]])
        # os.system(cmd)

        # # make individual plots for each transcript for full preds, ctrl preds, depr diff preds, labels
        # # plot ctrl preds
        # out_loc = output_loc + "/" + str(transcripts[worst_samples[i]]) + "/ctrl_preds.png"
        # plt.plot(ctrl_preds[worst_samples[i]], color='#2ecc71')
        # plt.title("Control Prediction", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.ylim([min_y, max_y])
        # plt.savefig(out_loc)
        # plt.clf()

        # # plot depr diff preds
        # out_loc = output_loc + "/" + str(transcripts[worst_samples[i]]) + "/depr_diff_preds.png"
        # plt.plot(depr_diffs[worst_samples[i]], color='#e74c3c')
        # plt.title("Deprivation Difference Prediction", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.ylim([min_y, max_y])
        # plt.savefig(out_loc)
        # plt.clf()

        # # plot full preds
        # out_loc = output_loc + "/" + str(transcripts[worst_samples[i]]) + "/full_preds.png"
        # plt.plot(full_preds[worst_samples[i]], color='#3498db')
        # plt.title("Full Prediction", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.ylim([min_y, max_y])
        # plt.savefig(out_loc)
        # plt.clf()

        # # plot labels
        # out_loc = output_loc + "/" + str(transcripts[worst_samples[i]]) + "/labels_full.png"
        # plt.plot(labels[worst_samples[i]], color='#f39c12')
        # plt.title("Full Labels", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.savefig(out_loc)
        # plt.clf()

        # # plot ctrl labels
        # out_loc = output_loc + "/" + str(transcripts[worst_samples[i]]) + "/labels_ctrl.png"
        # plt.plot(labels_ctrl[worst_samples[i]], color='#f39c12')
        # plt.title("Control Labels", fontsize=20)
        # plt.xlabel("Transcript Sequence", fontsize=20)
        # plt.ylabel("Normalized Read Counts", fontsize=20)
        # plt.savefig(out_loc)
        # plt.clf()





    