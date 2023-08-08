import pickle as pkl
from sklearn.model_selection import train_test_split

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final_presplit/VAL_feats.pkl', 'rb') as f:
    full_ds = pkl.load(f)

trascripts = list(full_ds.keys())
trascripts_train, transcripts_test = train_test_split(trascripts, test_size=0.2, random_state=42)
transcripts_train, transcripts_val = train_test_split(trascripts_train, test_size=0.25, random_state=42)

print("Train: ", len(transcripts_train))
print("Val: ", len(transcripts_val))
print("Test: ", len(transcripts_test))

# make the files
dict_train = {}
dict_val = {}
dict_test = {}

for i in range(len(transcripts_train)):
    dict_train[transcripts_train[i]] = full_ds[transcripts_train[i]]

for i in range(len(transcripts_val)):
    dict_val[transcripts_val[i]] = full_ds[transcripts_val[i]]

for i in range(len(transcripts_test)):
    dict_test[transcripts_test[i]] = full_ds[transcripts_test[i]]

# save these files
with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_VAL_train.pkl', 'wb') as f:
    pkl.dump(dict_train, f)

print("Saved train")

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_VAL_val.pkl', 'wb') as f:
    pkl.dump(dict_val, f)

print("Saved val")

with open('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/fin_VAL_test.pkl', 'wb') as f:
    pkl.dump(dict_test, f)

print("Saved test")