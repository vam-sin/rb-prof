# libraries
import pandas as pd 

# load in the data
ds = pd.read_csv('../data/rb_prof_Naef/processed_data/merged/merge_CTRL.csv')
ds.columns = ["index","gene", "transcript", "position_A_site", "count"]
ds['count_GScale'] = ds['count']/ds.groupby('gene')['count'].transform('sum')
ds = ds.drop(["index"], axis=1)
ds.to_csv('../data/rb_prof_Naef/processed_data/merge_norm/merge_CTRL_RIBO_gnorm.csv')
print(ds)