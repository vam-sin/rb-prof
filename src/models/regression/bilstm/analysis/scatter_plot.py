import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# af2
af2 = []

f = open('test_preds/af2.txt', 'r')
for line in f:
    af2.append(float(line.strip()))

af2 = np.array(af2)

# cbert
cbert = []

f = open('test_preds/cbert.txt', 'r')
for line in f:
    cbert.append(float(line.strip()))

cbert = np.array(cbert)

# cembeds
cembeds = []

f = open('test_preds/cembeds.txt', 'r')
for line in f:
    cembeds.append(float(line.strip()))

cembeds = np.array(cembeds)

# lem
lem = []

f = open('test_preds/lem.txt', 'r')
for line in f:
    lem.append(float(line.strip()))

lem = np.array(lem)

# mlm_cDNA_NT_IDAI
mlm_cDNA_NT_IDAI = []

f = open('test_preds/mlm_cdna_nt_idai.txt', 'r')
for line in f:
    mlm_cDNA_NT_IDAI.append(float(line.strip()))

mlm_cDNA_NT_IDAI = np.array(mlm_cDNA_NT_IDAI)

# mlm_cDNA_NT_pbert
mlm_cDNA_NT_pbert = []

f = open('test_preds/mlm_cdna_nt_pbert.txt', 'r')
for line in f:
    mlm_cDNA_NT_pbert.append(float(line.strip()))

mlm_cDNA_NT_pbert = np.array(mlm_cDNA_NT_pbert)

# nt
nt = []

f = open('test_preds/nt.txt', 'r')
for line in f:
    nt.append(float(line.strip()))

nt = np.array(nt)

# t5
t5 = []

f = open('test_preds/t5.txt', 'r')
for line in f:
    t5.append(float(line.strip()))

t5 = np.array(t5)

feature_preds = [af2, cbert, cembeds, lem, mlm_cDNA_NT_IDAI, mlm_cDNA_NT_pbert, nt, t5]
feature_names = ['af2', 'cbert', 'cembeds', 'lem', 'IDAI', 'PBERT', 'nt', 't5']

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = -0.1, 1
        low_y, high_y = -0.1, 1
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

# make scatter plot between af2 and t5
f, ax = plt.subplots(figsize=(10, 10))
ax.scatter(cbert, lem, s=10)
add_identity(ax, color='r', ls='--')
plt.show()
plt.xlabel('cbert')
plt.ylabel('lem')
plt.title('cbert vs lem')
plt.show()
plt.savefig('plots/SP_cbert_vs_lem.png')
plt.close()
