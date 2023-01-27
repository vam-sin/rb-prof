'''
gene-wise normalization of the count values
'''
# libraries
import pandas as pd 

# load in the data
ds = pd.read_csv('../data/rb_prof_Naef/AA_depr/CTRL_3_RIBO.tsv', sep=' ')
ds.columns = ["gene", "transcript", "position_A_site", "count"]
# ds.columns = ["index","gene", "transcript", "position_A_site", "count"]
ds['count_GScale'] = ds['count']/ds.groupby('gene')['count'].transform('sum')
# ds = ds.drop(["index"], axis=1)
ds.to_csv('../data/rb_prof_Naef/processed_proper/gnorm/CTRL_3_RIBO_gnorm.csv')
print(ds)