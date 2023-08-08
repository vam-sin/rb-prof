import numpy as np
from scipy.stats import pearsonr, spearmanr 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd

# nt_cbert
nt_cbert = {}
all_corr_list = []
all_files_list = []

f = open('ft_test_preds/nt_cbert.txt', 'r')
for line in f:
    line_spit = line.strip().split(' ')
    filename = line_spit[0].split('/')[-1]
    corr = float(line_spit[1])
    nt_cbert[filename] = corr 
    all_corr_list.append(corr)
    all_files_list.append(filename)
    # if corr > best_perf:
    #     best_perf = corr
    #     best_perf_sample = filename
    # if corr < worst_perf:
    #     worst_perf = corr
    #     worst_perf_sample = filename

# get lengths of best and worst 10% of predictions
file_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test/'

# load models preds
with open('ft_test_preds/nt_cbert_preds.pkl', 'rb') as f:
    nt_cbert_model_preds = pickle.load(f)

transcript_names = []

for x in all_files_list:
    transcript_names.append(x.split('_')[0])

transcript_names = list(set(transcript_names))

print(len(transcript_names), len(all_files_list))

list_mean_corr_between_y = []
list_mean_corrs_transcript = []
list_mean_corrs_between_preds = []

# for each transcript
count = 0
for transcript in transcript_names:
    print(transcript)
    count += 1
    if count % 100 == 0:
        print(count)
    y_list_for_transcript = []
    corr_list_for_transcript = []
    filenames_with_transcript = []
    preds_with_transcript = []
    for filename in all_files_list:
        if transcript in filename:
            full_file = file_path + filename
            corr_list_for_transcript.append(nt_cbert[filename])
            filenames_with_transcript.append(filename)
            preds_with_transcript.append(nt_cbert_model_preds[full_file])
            # open file and get y
            with open(file_path + filename, 'rb') as f:
                data_dict = pickle.load(f)
                y_list_for_transcript.append(data_dict['y'])
    
    print(transcript, len(y_list_for_transcript), len(corr_list_for_transcript), len(filenames_with_transcript))

    if len(y_list_for_transcript) > 1:
    
        corr_between_y = []
        corr_between_preds = []

        for i in range(len(y_list_for_transcript)):
            for j in range(i+1, len(y_list_for_transcript)):
                corr_between_y.append(pearsonr(y_list_for_transcript[i], y_list_for_transcript[j])[0])
                corr_between_preds.append(pearsonr(preds_with_transcript[i], preds_with_transcript[j])[0])

        # print(np.mean(corr_between_y))
        list_mean_corr_between_y.append(np.mean(corr_between_y)) # how similar are the y's in this transcript
        # print(np.mean(corr_list_for_transcript))
        list_mean_corrs_transcript.append(np.mean(corr_list_for_transcript)) # average performance of the model on this transcript
        # print(np.mean(corr_between_preds))
        list_mean_corrs_between_preds.append(np.mean(corr_between_preds)) # how similar are the predictions in this transcript
    # break

# print corr between lists
print(pearsonr(list_mean_corr_between_y, list_mean_corrs_transcript)) # relationship between similarity in y with performance: PearsonRResult(statistic=0.12845479584327937, pvalue=0.039611138696640354)
print(pearsonr(list_mean_corr_between_y, list_mean_corrs_between_preds)) # relationship between similarity in y with similarity in predictions: PearsonRResult(statistic=0.3072207417229852, pvalue=5.083893323055887e-07)
print(pearsonr(list_mean_corrs_transcript, list_mean_corrs_between_preds)) # relationship between performance and similarity in predictions: PearsonRResult(statistic=0.21759546008620909, pvalue=0.00044223059091901283)
print(np.mean(list_mean_corr_between_y), np.mean(list_mean_corrs_transcript), np.mean(list_mean_corrs_between_preds)) # 0.5794248522617663 0.48847529121066496 0.848435232269464

'''
list_mean_corr_between_y 0.5794248522617663 
list_mean_corrs_transcript 0.48847529121066496 
list_mean_corrs_between_preds 0.848435232269464

the predictions in each transcript are very similar to each other (0.848435232269464)
the true distributions in each transcript are not as similar to each other (0.5794248522617663)
'''

# # get second best and second worst and the filenames
# all_corr_list_copy = all_corr_list.copy()
# all_corr_list_copy.remove(best_perf)
# all_corr_list_copy.remove(worst_perf)
# second_best_perf = max(all_corr_list_copy)
# second_worst_perf = min(all_corr_list_copy)
# second_best_perf_sample = all_files_list[all_corr_list.index(second_best_perf)]
# second_worst_perf_sample = all_files_list[all_corr_list.index(second_worst_perf)]

# # third best and third worst
# all_corr_list_copy.remove(second_best_perf)
# all_corr_list_copy.remove(second_worst_perf)
# third_best_perf = max(all_corr_list_copy)
# third_worst_perf = min(all_corr_list_copy)
# third_best_perf_sample = all_files_list[all_corr_list.index(third_best_perf)]
# third_worst_perf_sample = all_files_list[all_corr_list.index(third_worst_perf)]

# # fourth best and fourth worst
# all_corr_list_copy.remove(third_best_perf)
# all_corr_list_copy.remove(third_worst_perf)
# fourth_best_perf = max(all_corr_list_copy)
# fourth_worst_perf = min(all_corr_list_copy)
# fourth_best_perf_sample = all_files_list[all_corr_list.index(fourth_best_perf)]
# fourth_worst_perf_sample = all_files_list[all_corr_list.index(fourth_worst_perf)]


# # print
# print(best_perf, best_perf_sample)
# print(worst_perf, worst_perf_sample)
# print(second_best_perf, second_best_perf_sample)
# print(second_worst_perf, second_worst_perf_sample)
# print(third_best_perf, third_best_perf_sample)
# print(third_worst_perf, third_worst_perf_sample)
# print(fourth_best_perf, fourth_best_perf_sample)
# print(fourth_worst_perf, fourth_worst_perf_sample)


# print(max(all_corr_list), best_perf, best_perf_sample)
# print(min(all_corr_list), worst_perf, worst_perf_sample)

# get filenames for best and worst 10% of predictions
# nt_cbert_sorted = sorted(nt_cbert.items(), key=lambda x: x[1])
# nt_cbert_best = nt_cbert_sorted[-int(len(nt_cbert_sorted)*0.1):]
# nt_cbert_worst = nt_cbert_sorted[:int(len(nt_cbert_sorted)*0.1)]

# # get best performing sample
# best_filename = nt_cbert_best[1][0]
# print(best_filename)
# # print performance
# print(nt_cbert_best[1][1])

# get best and worst 10% of predictions
# nt_cbert_best_preds = [x[1] for x in nt_cbert_best]
# nt_cbert_worst_preds = [x[1] for x in nt_cbert_worst]

# # # plot for best
# best_filename = nt_cbert_best[1][0]
# print(best_filename)
# full_path = file_path + best_filename
# model_preds_best_sample = nt_cbert_model_preds[full_path]


# # normalize preds
# model_preds_best_sample = (model_preds_best_sample - np.min(model_preds_best_sample)) / (np.max(model_preds_best_sample) - np.min(model_preds_best_sample))
# with open(full_path, 'rb') as f:
#     data = pickle.load(f)
#     y = data['y']

# # normalize y
# y = (y - np.min(y)) / (np.max(y) - np.min(y))
    
# # plot y and model preds best sample 
# df = pd.DataFrame({'Predicted':model_preds_best_sample,'True':y})
# g = sns.lineplot(data = df, palette = ['#f1c40f', '#3498db'], dashes=False)
# # g.axhline(0.0)
# # sns.lineplot(x=len_lis, y=y_pred_imp_seq_neg, label='Predicted', color='#f0932b')
# # sns.lineplot(x=len_lis, y=y_true_imp_seq, label='True', color='#6ab04c')
# # sns.despine(left=True, bottom=True)
# # g.set(xticklabels=[], yticklabels=[])  # remove the tick labels
# # g.tick_params(bottom=False, left=False)  # remove the ticks
# plt.xlabel("Gene Sequence")
# plt.ylabel("Normalized Ribosome Counts")
# plt.legend()
# plt.show()
# out_path = 'bw_plots/worst_samples_other_conds/nt_cbert_worst_other_conds_sample_' + best_filename + '.png'
# plt.savefig(out_path)
# plt.clf()

# # plot for worst
# worst_filename = nt_cbert_worst[5][0]
# print(worst_filename)
# full_path = file_path + worst_filename
# model_preds_worst_sample = nt_cbert_model_preds[full_path]
# # normalize preds
# model_preds_worst_sample = (model_preds_worst_sample - np.min(model_preds_worst_sample)) / (np.max(model_preds_worst_sample) - np.min(model_preds_worst_sample))
# with open(full_path, 'rb') as f:
#     data = pickle.load(f)
#     y = data['y']

# # normalize y
# print(y)
# y = (y - np.min(y)) / (np.max(y) - np.min(y))
# print(y)
# # plot y and model preds best sample 
# df = pd.DataFrame({'Predicted':model_preds_worst_sample,'True':y})
# g = sns.lineplot(data = df, palette = ['#f1c40f', '#3498db'], dashes=False)
# # g.axhline(0.0)
# # sns.lineplot(x=len_lis, y=y_pred_imp_seq_neg, label='Predicted', color='#f0932b')
# # sns.lineplot(x=len_lis, y=y_true_imp_seq, label='True', color='#6ab04c')
# # sns.despine(left=True, bottom=True)
# # g.set(xticklabels=[], yticklabels=[])  # remove the tick labels
# # g.tick_params(bottom=False, left=False)  # remove the ticks
# plt.xlabel("Gene Sequence")
# plt.ylabel("Normalized Ribosome Counts")
# plt.legend()
# plt.show()
# out_path = 'bw_plots/nt_cbert_worst_sample_' + worst_filename + '.png'
# plt.savefig(out_path)

# Sequence Lengths and Perc Annots Analysis
# nt_cbert_best_lengths = []

# for filename, _ in nt_cbert_best:
#     full_path = file_path + filename
#     with open(full_path, 'rb') as f:
#         data = pickle.load(f)
#     nt_cbert_best_lengths.append(len(data['sequence']))

# nt_cbert_worst_lengths = []

# for filename, _ in nt_cbert_worst:
#     full_path = file_path + filename
#     with open(full_path, 'rb') as f:
#         data = pickle.load(f)
#     nt_cbert_worst_lengths.append(len(data['sequence']))


# # get perc annots for best and worst 10% of predictions
# nt_cbert_best_perc_annots = []

# for filename, _ in nt_cbert_best:
#     full_path = file_path + filename
#     with open(full_path, 'rb') as f:
#         data = pickle.load(f)
#         len_ = len(data['sequence'])
#         y = data['y']
#         num_non_zero = np.count_nonzero(y)
#         perc_annot = num_non_zero / len_
#     nt_cbert_best_perc_annots.append(perc_annot)

# nt_cbert_worst_perc_annots = []

# for filename, _ in nt_cbert_worst:
#     full_path = file_path + filename
#     with open(full_path, 'rb') as f:
#         data = pickle.load(f)
#         len_ = len(data['sequence'])
#         y = data['y']
#         num_non_zero = np.count_nonzero(y)
#         perc_annot = num_non_zero / len_
#     nt_cbert_worst_perc_annots.append(perc_annot)

# print("Best Preds")
# print(nt_cbert_best)
# print(nt_cbert_best_lengths)
# print("Mean Corr of Best", np.mean(nt_cbert_best_preds))
# print("Mean Length of Best", np.mean(nt_cbert_best_lengths))
# print("Mean Perc Annot of Best", np.mean(nt_cbert_best_perc_annots))
# print("Worst Preds")
# print(nt_cbert_worst)
# print(nt_cbert_worst_lengths)
# print("Mean Corr of Worst", np.mean(nt_cbert_worst_preds))
# print("Mean Length of Worst", np.mean(nt_cbert_worst_lengths))
# print("Mean Perc Annot of Worst", np.mean(nt_cbert_worst_perc_annots))

# # 1d scatter plot of best and worst predictions lengths tagged by color
# plt.figure(figsize=(10, 10))
# plt.scatter(nt_cbert_best_lengths, nt_cbert_best_preds, c='orange', label='Best 10%')
# plt.scatter(nt_cbert_worst_lengths, nt_cbert_worst_preds, c='blue', label='Worst 10%')
# plt.xlabel('Length')
# plt.ylabel('Correlation')
# plt.legend()
# plt.savefig('bw_plots/best_and_worst_lengths_nt_cbert.png')
# plt.close()

# # 1d scatter plot of best and worst perc_annots tagged by color
# plt.figure(figsize=(10, 10))
# plt.scatter(nt_cbert_best_perc_annots, nt_cbert_best_preds, c='orange', label='Best 10%')
# plt.scatter(nt_cbert_worst_perc_annots, nt_cbert_worst_preds, c='blue', label='Worst 10%')
# plt.xlabel('Perc Annotated')
# plt.ylabel('Correlation')
# plt.legend()
# plt.savefig('bw_plots/best_and_worst_perc_annots_nt_cbert.png')
# plt.close()

# # get lengths of all the sequences
# all_lengths = []
# for filename in all_files_list:
#     full_path = file_path + filename
#     with open(full_path, 'rb') as f:
#         data = pickle.load(f)
#     all_lengths.append(len(data['sequence']))

# # get corr between lengths and corr
# print("Pearson Corr", pearsonr(all_lengths, all_corr_list))
# print("Spearman Corr", spearmanr(all_lengths, all_corr_list))

# # make 2d scatter plot of lengths and corr with log of length
# all_lengths = np.log(all_lengths)
# plt.figure(figsize=(10, 10))
# plt.scatter(all_lengths, all_corr_list)
# plt.xlabel('Log Length')
# plt.ylabel('Correlation')
# plt.savefig('bw_plots/all_lengthsLOG_vs_corr.png')
# plt.close()

# # # get percentage annotated for all the sequences
# all_perc_annotated = []
# for filename in all_files_list:
#     full_path = file_path + filename
#     with open(full_path, 'rb') as f:
#         data = pickle.load(f)
#         len_ = len(data['sequence'])
#         y = data['y']
#         # get num non zero from y 
#         num_non_zero = np.count_nonzero(y)
#         perc_annotated = num_non_zero/len_ 
#     all_perc_annotated.append(perc_annotated)

# # # get corr between perc_annotated and corr
# print("Pearson Corr", pearsonr(all_perc_annotated, all_corr_list))
# print("Spearman Corr", spearmanr(all_perc_annotated, all_corr_list))

# # make 2d scatter plot of perc_annotated and corr
# plt.figure(figsize=(10, 10))
# plt.scatter(all_perc_annotated, all_corr_list)
# plt.xlabel('Percentage Annotated')
# plt.ylabel('Correlation')
# plt.savefig('bw_plots/all_perc_annotated_vs_corr.png')
# plt.close()

# make a heatmap with length and perc_annotated with color of corr
# all_lengths = np.log(all_lengths)
# plt.figure(figsize=(10, 10))
# plt.scatter(all_lengths, all_perc_annotated, c=all_corr_list)
# plt.xlabel('Log Length')
# plt.ylabel('Percentage Annotated')
# plt.colorbar()
# plt.savefig('bw_plots/LOG_all_lengths_vs_perc_annotated_withCorr.png')
# plt.close()

# best_filename = 'ENSMUST00000033583.13_LEU_.pkl'
# print(best_filename)
# full_path = file_path + best_filename
# leu_preds = nt_cbert_model_preds[full_path]

# # ile y
# with open(full_path, 'rb') as f:
#     data = pickle.load(f)
#     leu_y = data['y']

# best_filename = 'ENSMUST00000033583.13_LEU-ILE_.pkl'
# print(best_filename)
# full_path = file_path + best_filename
# ctrl_preds = nt_cbert_model_preds[full_path]

# # ctrl y
# with open(full_path, 'rb') as f:
#     data = pickle.load(f)
#     ctrl_y = data['y']

# best_filename = 'ENSMUST00000033583.13_ILE_.pkl'
# print(best_filename)
# full_path = file_path + best_filename
# ile_preds = nt_cbert_model_preds[full_path]

# # ile y
# with open(full_path, 'rb') as f:
#     data = pickle.load(f)
#     ile_y = data['y']


# # performances
# print("LEU", pearsonr(leu_preds, leu_y))
# print("LEU-ILE", pearsonr(ctrl_preds, ctrl_y))
# print("ILE", pearsonr(ile_preds, ile_y))

# # print corrs between leu and ctrl
# print("Pearson Corr preds LEU-(LEU-ILE)", pearsonr(leu_preds, ctrl_preds))
# # corr between the y's
# print("Pearson Corr y LEU-(LEU-ILE)", pearsonr(leu_y, ctrl_y))

# # print corrs between leu and ile
# print("Pearson Corr preds LEU-ILE", pearsonr(leu_preds, ile_preds))
# # corr between the y's
# print("Pearson Corr y LEU-ILE", pearsonr(leu_y, ile_y))

# # print corrs between ctrl and ile
# print("Pearson Corr preds (LEU-ILE)-ILE", pearsonr(ctrl_preds, ile_preds))
# # corr between the y's
# print("Pearson Corr y (LEU-ILE)-ILE", pearsonr(ctrl_y, ile_y))

''' BEST and WORST 10%
------------ Best ------------
Mean Corr of Best 0.7035760264009926
Mean Length of Best 232.20192307692307
Mean Perc Annot of Best 0.7758749108595933

------------ Worst ------------
Mean Corr of Worst 0.27420675026157065
Mean Length of Worst 295.88461538461536
Mean Perc Annot of Worst 0.7230031839353568
'''

''' worst sample: ENSMUST00000084125.9
LEU PearsonRResult(statistic=0.21337611283803204, pvalue=1.163679363219743e-05)
CTRL PearsonRResult(statistic=0.2039100693855042, pvalue=2.8455076690928725e-05)
ILE PearsonRResult(statistic=0.09042227488301458, pvalue=0.06573050013889917)
Pearson Corr preds LEU-CTRL PearsonRResult(statistic=0.9725707296997735, pvalue=1.0500096927986162e-263)
Pearson Corr y LEU-CTRL PearsonRResult(statistic=0.7166048514155842, pvalue=1.3058515930208495e-66)
Pearson Corr preds LEU-ILE PearsonRResult(statistic=0.8937794249320656, pvalue=6.6498886296851485e-146)
Pearson Corr y LEU-ILE PearsonRResult(statistic=0.6271226294522604, pvalue=9.581277920909286e-47)
Pearson Corr preds CTRL-ILE PearsonRResult(statistic=0.8981502519991457, pvalue=1.8159354205509908e-149)
Pearson Corr y CTRL-ILE PearsonRResult(statistic=0.6046845829466807, pvalue=9.822677583228972e-43)
mean of preds corr: 0.9215001355436616
mean of y corr: 0.6494700216048418
'''

''' third worst sample: ENSMUST00000112571.4_CTRL_.pkl
LEU PearsonRResult(statistic=0.3563926928514777, pvalue=0.00027328968963847287)
CTRL PearsonRResult(statistic=0.10336492257835454, pvalue=0.30611969239123776)
ILE PearsonRResult(statistic=0.19040526394947505, pvalue=0.05776037200215815)
Pearson Corr preds LEU-CTRL PearsonRResult(statistic=0.8275447151438051, pvalue=2.594970508699387e-26)
Pearson Corr y LEU-CTRL PearsonRResult(statistic=0.7738384788354359, pvalue=3.77582710013575e-21)
Pearson Corr preds LEU-ILE PearsonRResult(statistic=0.762086348218949, pvalue=3.311421390413699e-20)
Pearson Corr y LEU-ILE PearsonRResult(statistic=0.7088453949477598, pvalue=1.566613937657161e-16)
Pearson Corr preds CTRL-ILE PearsonRResult(statistic=0.8873921481236052, pvalue=1.000113101526778e-34)
Pearson Corr y CTRL-ILE PearsonRResult(statistic=0.7318876345089569, pvalue=5.153209975673954e-18)
mean of preds corr: 0.8256744034954534
mean of y corr: 0.7388575027643841
'''

''' fourth worst sample: ENSMUST00000033583.13_LEU-ILE_.pkl
LEU PearsonRResult(statistic=0.21204942730277818, pvalue=4.0203444593048785e-05)
LEU-ILE PearsonRResult(statistic=0.11088878196672082, pvalue=0.03321720546942863)
ILE PearsonRResult(statistic=0.30152739610128987, pvalue=3.408109809675281e-09)
Pearson Corr preds LEU-(LEU-ILE) PearsonRResult(statistic=0.9087332571870811, pvalue=2.4815993726825337e-141)
Pearson Corr y LEU-(LEU-ILE) PearsonRResult(statistic=0.8278492912539753, pvalue=3.598730543156632e-94)
Pearson Corr preds LEU-ILE PearsonRResult(statistic=0.6658788806825108, pvalue=1.265871171177707e-48)
Pearson Corr y LEU-ILE PearsonRResult(statistic=0.8213119489925115, pvalue=1.7550491925373087e-91)
Pearson Corr preds (LEU-ILE)-ILE PearsonRResult(statistic=0.8992406225128338, pvalue=7.717386891390808e-134)
Pearson Corr y (LEU-ILE)-ILE PearsonRResult(statistic=0.6703921460698005, pvalue=1.7032816456526902e-49)
mean of preds corr: 0.8259505867948085
mean of y corr: 0.773501795772765
'''

''' second best sample: ENSMUST00000030905.8
LEU PearsonRResult(statistic=0.8299124992582426, pvalue=8.374880165003875e-51)
CTRL PearsonRResult(statistic=0.6480263576976312, pvalue=1.3159864103109963e-24)
ILE PearsonRResult(statistic=0.8731474681597343, pvalue=3.875983179739147e-62)
Pearson Corr preds LEU-CTRL PearsonRResult(statistic=0.9040571214520343, pvalue=3.591317584374673e-73)
Pearson Corr y LEU-CTRL PearsonRResult(statistic=0.785814245674546, pvalue=3.849484372608732e-42)
Pearson Corr preds LEU-ILE PearsonRResult(statistic=0.9038501156475367, pvalue=4.37644593161664e-73)
Pearson Corr y LEU-ILE PearsonRResult(statistic=0.8654027710403267, pvalue=7.983257236795286e-60)
Pearson Corr preds CTRL-ILE PearsonRResult(statistic=0.8817569465706712, pvalue=6.773984791949329e-65)
Pearson Corr y CTRL-ILE PearsonRResult(statistic=0.7932540790822465, pvalue=1.8792560973279263e-43)
mean of preds corr: 0.8959227272239447
mean of y corr: 0.8103170543596816
'''

''' third best sample: ENSMUST00000112172.3
LEU PearsonRResult(statistic=0.8643246099647639, pvalue=2.0313235943212433e-14)
CTRL PearsonRResult(statistic=0.7866420633418931, pvalue=1.498951469780949e-10)
ILE PearsonRResult(statistic=0.6986822193824005, pvalue=9.445688208182204e-08)
Pearson Corr preds LEU-CTRL PearsonRResult(statistic=0.970265357029153, pvalue=4.005180560441401e-28)
Pearson Corr y LEU-CTRL PearsonRResult(statistic=0.9897814645466531, pvalue=5.165513350572767e-38)
Pearson Corr preds LEU-ILE PearsonRResult(statistic=0.8144818633086401, pvalue=1.0011885891420983e-11)
Pearson Corr y LEU-ILE PearsonRResult(statistic=0.9808196199935681, pvalue=3.5835009166738204e-32)
Pearson Corr preds CTRL-ILE PearsonRResult(statistic=0.7937716227881343, pvalue=7.797761967871245e-11)
Pearson Corr y CTRL-ILE PearsonRResult(statistic=0.972945280048036, pvalue=5.39819884967572e-29)
mean of preds corr: 0.8595069470416428
mean of y corr: 0.9144639015299587
'''

''' fourth best sample: ENSMUST00000112172.3_VAL_.pkl
LEU PearsonRResult(statistic=0.8643246099647639, pvalue=2.0313235943212433e-14)
VAL PearsonRResult(statistic=0.8458792142812844, pvalue=2.5941764909711537e-13)
ILE PearsonRResult(statistic=0.6986822193824005, pvalue=9.445688208182204e-08)
Pearson Corr preds LEU-VAL PearsonRResult(statistic=0.9931242245315176, pvalue=1.0664700773739279e-41)
Pearson Corr y LEU-VAL PearsonRResult(statistic=0.9977110085983677, pvalue=5.990095198351995e-52)
Pearson Corr preds LEU-ILE PearsonRResult(statistic=0.8144818633086401, pvalue=1.0011885891420983e-11)
Pearson Corr y LEU-ILE PearsonRResult(statistic=0.9808196199935681, pvalue=3.5835009166738204e-32)
Pearson Corr preds VAL-ILE PearsonRResult(statistic=0.8345118878906872, pvalue=1.062457864814516e-12)
Pearson Corr y VAL-ILE PearsonRResult(statistic=0.9783503689203176, pvalue=4.725639331797527e-31)
mean of preds corr: 0.8800399919102316
mean of y corr: 0.9529609995047271
'''

'''
------------ Length vs Pearson Corr ------------
Pearson Corr PearsonRResult(statistic=-0.09675226098815873, pvalue=0.0017494842242955953)
Spearman Corr SignificanceResult(statistic=-0.13919299726953197, pvalue=6.359617291293031e-06)

------------ Perc Annotated vs Corr ------------
Pearson Corr PearsonRResult(statistic=0.21345041004017212, pvalue=3.1920583112956633e-12)
Spearman Corr SignificanceResult(statistic=0.22035508942306078, pvalue=6.019424671870593e-13)
'''

'''
hypothesis: those samples that have their distributions pretty similar over the conditions have better predictions, whereas those that have very different distributions have worse predictions

worst
mean of preds corr: 0.9215001355436616
mean of y corr: 0.6494700216048418

mean of preds corr: 0.8256744034954534
mean of y corr: 0.7388575027643841

mean of preds corr: 0.8259505867948085
mean of y corr: 0.773501795772765

--- best

mean of preds corr: 0.8959227272239447
mean of y corr: 0.8103170543596816

mean of preds corr: 0.8595069470416428
mean of y corr: 0.9144639015299587

mean of preds corr: 0.8800399919102316
mean of y corr: 0.9529609995047271

'''