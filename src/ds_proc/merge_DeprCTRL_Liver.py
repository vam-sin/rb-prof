import pandas as pd 

# Read in the data
liver_data = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/liver.csv'
depr_ctrl_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/CTRL.csv'

liver_df = pd.read_csv(liver_data)
liver_df.columns = ['gene', 'sequence', 'annotations', 'perc_non_zero_annots']
depr_ctrl_df = pd.read_csv(depr_ctrl_path)

# redo the calculations for perc_non_zero_annots
annotations_depr_ctrl = list(depr_ctrl_df['annotations'])

redid_perc_non_zero_annots = []

for annot in annotations_depr_ctrl:
    annot = annot[1:-1].split(', ')
    annot = [float(i) for i in annot]
    annot = annot[5:-5]
    num_non_zero = sum([1 for i in annot if i != 0.0])
    redid_perc_non_zero_annots.append(num_non_zero/len(annot))

depr_ctrl_df['perc_non_zero_annots'] = redid_perc_non_zero_annots

# redo the calculations for perc_non_zero_annots in liver_df
annotations_liver = list(liver_df['annotations'])

redid_perc_non_zero_annots = []

for annot in annotations_liver:
    annot = annot[1:-1].split(', ')
    annot = [float(i) for i in annot]
    annot = annot[5:-5]
    num_non_zero = sum([1 for i in annot if i != 0.0])
    redid_perc_non_zero_annots.append(num_non_zero/len(annot))

liver_df['perc_non_zero_annots'] = redid_perc_non_zero_annots

print(liver_df)
print(depr_ctrl_df)

# find common genes between the two dataframes
common_genes = set(liver_df['gene']).intersection(set(depr_ctrl_df['gene']))

print(len(set(list(liver_df['gene']))))
print(len(set(list(depr_ctrl_df['gene']))))

print(len(common_genes))

# add those samples from depr_ctrl_df to liver_df which are not in common genes
depr_ctrl_df = depr_ctrl_df[~depr_ctrl_df['gene'].isin(common_genes)]
print(depr_ctrl_df)

# concat the two dataframes
full_df = pd.concat([liver_df, depr_ctrl_df], ignore_index=True)

print(full_df)

full_df.to_csv('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/liver_deprCTRL.csv', index=False)
