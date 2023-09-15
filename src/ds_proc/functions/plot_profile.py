import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pywt
from scipy import stats

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    print(m, sd)
    return np.where(sd == 0, 0, m/sd)

def make_plot(y, filename):
    y = y 
    
    df = pd.DataFrame({'CTRL':y})
    g = sns.lineplot(data = df, palette = ['#2ecc71'], dashes=False)
    plt.xlabel("Gene Sequence")
    plt.ylabel("Normalized Ribosome Counts")
    plt.legend()
    plt.show()
    plt.savefig(filename, format="png")


df = pd.read_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/liver.csv')

print(df)

sample_num = 10

# apply 30% cutoff
df = df[df['perc_non_zero_annots'] > 0.3]

annots = list(df['annotations'])

y = annots[sample_num]

y = y[1:-1].split(', ')
y = [float(i) for i in y]

# min max signal
min_max_y = [(i - min(y)) / (max(y) - min(y)) for i in y]

# log_signal
log_signal = [1+i for i in y]
log_signal = np.log(log_signal)

# denoised signal
wavelet = 'haar'
cA, cD = pywt.dwt(y, wavelet)
only_approx = pywt.idwt(cA, None, wavelet)
only_details = pywt.idwt(None, cD, wavelet)
only_approx = np.asarray(only_approx)
only_details = np.asarray(only_details)
y = np.asarray(y)

# log approx signal
log_approx = [1+i for i in only_approx]
log_approx = np.log(log_approx)
# print(y.shape, only_approx.shape, only_details.shape)

out_file = "plots/liver_{}.png".format(sample_num)
out_min_max_file = "plots/liver_{}_min_max.png".format(sample_num)
out_log_file = "plots/liver_{}_log.png".format(sample_num)
out_wt_approx = "plots/liver_{}_wt_approx.png".format(sample_num)
out_wt_details = "plots/liver_{}_wt_details.png".format(sample_num)
out_wt_approx_log = "plots/liver_{}_wt_approx_log.png".format(sample_num)

full_matrix = [y, min_max_y, log_signal, only_approx, only_details, log_approx]

make_plot(y, out_file)
plt.clf()
make_plot(min_max_y, out_min_max_file)
plt.clf()
make_plot(log_signal, out_log_file)
plt.clf()
make_plot(only_approx, out_wt_approx)
plt.clf()
make_plot(only_details, out_wt_details)
plt.clf()
make_plot(log_approx, out_wt_approx_log)
plt.clf() 

full_matrix = np.asarray(full_matrix)
print(full_matrix.shape)

print("SNR: ", signaltonoise(y))
