'''
gene-wise normalization of the count values
'''
# libraries
import pandas as pd 
from os import listdir
from os.path import isfile, join

folder_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/liver_samples/raw/'
files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
files = [f.replace('.tsv', '') for f in files]

print(files)

for file in files:
    in_file = folder_path + file + '.tsv'
    out_file = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/liver_samples/gene_norm/' + file + '.csv'

    # # load in the data
    ds = pd.read_csv(in_file, sep=' ')
    ds.columns = ["gene", "transcript", "position_A_site", "count"]
    # ds.columns = ["index","gene", "transcript", "position_A_site", "count"]
    ds['count_GScale'] = ds['count']/ds.groupby('gene')['count'].transform('sum')
    # ds = ds.drop(["index"], axis=1)
    ds.to_csv(out_file)
    # print(ds)