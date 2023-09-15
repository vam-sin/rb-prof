import numpy as np
from scipy.stats import pearsonr, spearmanr 
import matplotlib.pyplot as plt
import seaborn as sns

# af2
af2 = []

f = open('ft_test_preds/af2.txt', 'r')
for line in f:
    af2.append(float(line.strip()))

af2 = np.array(af2)

# cbert
cbert = []

f = open('ft_test_preds/cbert.txt', 'r')
for line in f:
    cbert.append(float(line.strip()))

cbert = np.array(cbert)

# cembeds
cembeds = []

f = open('ft_test_preds/cembeds.txt', 'r')
for line in f:
    cembeds.append(float(line.strip()))

cembeds = np.array(cembeds)

# lem
lem = []

f = open('ft_test_preds/lem.txt', 'r')
for line in f:
    lem.append(float(line.strip()))

lem = np.array(lem)

# mlm_cDNA_NT_IDAI
mlm_cDNA_NT_IDAI = []

f = open('ft_test_preds/mlm_cdna_nt_idai.txt', 'r')
for line in f:
    mlm_cDNA_NT_IDAI.append(float(line.strip()))

mlm_cDNA_NT_IDAI = np.array(mlm_cDNA_NT_IDAI)

# mlm_cDNA_NT_pbert
mlm_cDNA_NT_pbert = []

f = open('ft_test_preds/mlm_cdna_nt_pbert.txt', 'r')
for line in f:
    mlm_cDNA_NT_pbert.append(float(line.strip()))

mlm_cDNA_NT_pbert = np.array(mlm_cDNA_NT_pbert)

# nt
nt = []

f = open('ft_test_preds/nt.txt', 'r')
for line in f:
    nt.append(float(line.strip()))

nt = np.array(nt)

# t5
t5 = []

f = open('ft_test_preds/t5.txt', 'r')
for line in f:
    t5.append(float(line.strip()))

t5 = np.array(t5)

# geom
geom = []

f = open('ft_test_preds/geom.txt', 'r')
for line in f:
    geom.append(float(line.strip()))

geom = np.array(geom)

print(af2.shape, cbert.shape, cembeds.shape, lem.shape, mlm_cDNA_NT_IDAI.shape, mlm_cDNA_NT_pbert.shape, nt.shape, t5.shape, geom.shape)

feature_preds = [af2, cbert, cembeds, lem, mlm_cDNA_NT_IDAI, mlm_cDNA_NT_pbert, nt, t5, geom]
feature_names = ['AF2-SS', 'Codon DE', 'Codon PEL', 'RNA-SS-LEV', 'Codon NTrans', 'Codon BERT', 'NE', 'T5', 'Geom']

# 9 x 9 heatmap
corr_mat = np.zeros((len(feature_preds), len(feature_preds)))


for i in range(len(feature_preds)):
    for j in range(len(feature_preds)):
        corr = pearsonr(feature_preds[i], feature_preds[j])[0]
        corr_mat[i][j] = corr

print(corr_mat)

# make seaborn heatmap from crr_mat
# ax = sns.heatmap(corr_mat, annot=True, xticklabels=feature_names, yticklabels=feature_names, cmap='Blues')
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")
# plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
#             rotation_mode="anchor")
# sns cluster map
g = sns.clustermap(corr_mat, annot=True, xticklabels=feature_names, yticklabels=feature_names, figsize=(10, 10))
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 12, rotation=45, ha="right", rotation_mode="anchor")
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 12, rotation=0, ha="left", rotation_mode="anchor")
# g.ax_heatmap.set_xlabel('Features', fontsize=14)
# g.ax_heatmap.set_ylabel('Features', fontsize=14)
# g.ax_heatmap.set_title('Correlation Heatmap of Features', fontsize=16)

plt.tight_layout()
plt.savefig('ablation_plots/9_ft_corr_heatmap_clustered.png', dpi=300)
plt.show()