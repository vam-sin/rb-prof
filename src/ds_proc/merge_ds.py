'''
merges multiple dataframes of a certain condition and averages the count values for them.
'''
# libraries
import pandas as pd 
import numpy as np 

def mean_string(vals):
    vals_split = vals.split(',')
    # print(vals_split)
    vals_split = [float(x) for x in vals_split]
    # print(vals_split)
    return float(np.mean(vals_split))

ds1 = pd.read_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr/CTRL_1_RIBO.tsv', sep=' ')
ds1.columns = ["gene", "transcript", "position_A_site", "count"]
# ds1 = ds1.drop(["index"], axis=1)
print(ds1)

ds2 = pd.read_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr/CTRL_2_RIBO.tsv', sep=' ')
ds2.columns = ["gene", "transcript", "position_A_site", "count"]
# ds2 = ds2.drop(["index"], axis=1)
# print(ds)

ds3 = pd.read_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr/CTRL_3_RIBO.tsv', sep=' ')
ds3.columns = ["gene", "transcript", "position_A_site", "count"]
# ds3 = ds3.drop(["index"], axis=1)
# print(ds)

# merge the dataframes
merge_df = pd.DataFrame(columns=["gene", "transcript", "position_A_site", "count"])

print(merge_df)

tr_1 = list(set(list(ds1["transcript"])))
tr_2 = list(set(list(ds2["transcript"])))
tr_3 = list(set(list(ds3["transcript"])))

tr_full = list(set(tr_1 + tr_2 + tr_3))

# print(len(tr_full), len(tr_1), len(tr_2), len(tr_3))

for i in range(len(tr_full)):
    print(i, len(tr_full))
    ds1_seg = ds1[ds1["transcript"].isin([tr_full[i]])]
    ds2_seg = ds2[ds2["transcript"].isin([tr_full[i]])]
    ds3_seg = ds3[ds3["transcript"].isin([tr_full[i]])]
    ds_seg_merge = pd.concat([ds1_seg, ds2_seg, ds3_seg], axis=0)
    gene_id = list(ds_seg_merge["gene"])[0]
    ds_seg_merge['count'] = ds_seg_merge['count'].astype(str)
    ds_seg_merge.groupby('position_A_site')['count'].agg(','.join).reset_index()
    # print(ds_seg_merge)
    ds_seg_merge["count"] = ds_seg_merge["count"].apply(mean_string)
    # print(ds_seg_merge)
    merge_df = pd.concat([merge_df, ds_seg_merge], axis=0)
    # print(merge_df)

merge_df.to_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/merge_NoNorm/merge_NoNorm_CTRL.csv')