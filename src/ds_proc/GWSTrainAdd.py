'''
conducts gene wise split for the dataset and splits the common genes and ups them in one of the sets
'''
from os import listdir
from os.path import isfile, join
import os
from collections import Counter

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/train'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
train_files = [mypath + '/' + f for f in onlyfiles]

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/val'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
val_files = [mypath + '/' + f for f in onlyfiles]

mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/test'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
test_files = [mypath + '/' + f for f in onlyfiles]

print("Train: ", len(train_files), " Val: ", len(val_files), " Test: ", len(test_files))

train_genes = [f.split('/')[-1].split('_')[0] for f in train_files]
val_genes = [f.split('/')[-1].split('_')[0] for f in val_files]
test_genes = [f.split('/')[-1].split('_')[0] for f in test_files]

print(train_files[0], train_genes[0])


def remove_common_genes(train_files, val_files, train_genes, val_genes):
    # remove common genes between train and val (remove them from train and put them in val)
    new_train_files = train_files.copy()
    tr_val_common = [x for x in train_genes if x in val_genes]
    tr_val_common = list(set(tr_val_common))
    # split common files into two
    tr_val_common_1 = tr_val_common[:len(tr_val_common)//2]
    tr_val_common_2 = tr_val_common[len(tr_val_common)//2:]
    print("Common genes between the two lists: ", len(tr_val_common))
    for i in range(len(tr_val_common_1)):
        # take those gene files and put them in val
        for j in range(len(train_files)):
            # print(train_files[j].split('/')[-1].split('_')[0], tr_val_common[i])
            if train_files[j].split('/')[-1].split('_')[0] == tr_val_common_1[i]:
                val_files.append(train_files[j])
                new_train_files.remove(train_files[j])

    new_val_files = val_files.copy()
    for i in range(len(tr_val_common_2)):
        # take those gene files and put them in val
        for j in range(len(val_files)):
            # print(train_files[j].split('/')[-1].split('_')[0], tr_val_common[i])
            if val_files[j].split('/')[-1].split('_')[0] == tr_val_common_2[i]:
                new_train_files.append(val_files[j])
                new_val_files.remove(val_files[j])

    return new_train_files, new_val_files

def get_num_common_genes(train_files, val_files, test_files):
    train_genes = [f.split('/')[-1].split('_')[0] for f in train_files]
    val_genes = [f.split('/')[-1].split('_')[0] for f in val_files]
    test_genes = [f.split('/')[-1].split('_')[0] for f in test_files]
    tr_val_common = [x for x in train_genes if x in val_genes]
    tr_test_common = [x for x in train_genes if x in test_genes]
    val_test_common = [x for x in val_genes if x in test_genes]
    print("Common genes between train and val: ", len(list(set(tr_val_common))))
    print("Common genes between train and test: ", len(list(set(tr_test_common))))
    print("Common genes between val and test: ", len(list(set(val_test_common))))

def get_condition_split(train_files, val_files, test_files):
    print("----- Condition Split -----")
    # get the number of files per condition
    train_conditions = [f.split('/')[-1].split('_')[1] for f in train_files]
    val_conditions = [f.split('/')[-1].split('_')[1] for f in val_files]
    test_conditions = [f.split('/')[-1].split('_')[1] for f in test_files]
    print("Train: ", Counter(train_conditions))
    print("Val: ", Counter(val_conditions))
    print("Test: ", Counter(test_conditions))

# get num common
get_num_common_genes(train_files, val_files, test_files)
print("Initial Condition Split")
get_condition_split(train_files, val_files, test_files)

# def add_some_genes_to_val(train_genes, test_genes, train_files, test_files):
#     # add half of test genes to train
#     test_genes_unique = list(set(test_genes))
#     test_genes_1 = test_genes_unique[:len(test_genes_unique)//2] # ones being removed from test
#     test_genes_2 = test_genes_unique[len(test_genes_unique)//2:]
#     print("Test genes Unique: ", len(test_genes_unique))
#     print("Test genes 1: ", len(test_genes_1))
#     print("Test genes 2: ", len(test_genes_2))

#     test_files_being_moved = []
#     for i in range(len(test_genes_1)):
#         for j in range(len(test_files)):
#             if test_files[j].split('/')[-1].split('_')[0] == test_genes_1[i]:
#                 test_files_being_moved.append(test_files[j])
    
#     print("Number of test files being moved: ", len(test_files_being_moved))
#     print(test_files_being_moved[0])

#     # move the test files to train
#     for i in range(len(test_files_being_moved)):
#         print(i)
#         old_path = test_files_being_moved[i]
#         new_filename = old_path.replace('val', 'train')
#         cmd = 'mv ' + old_path + ' ' + new_filename
#         os.system(cmd)

# # add_some_genes_to_val(train_genes, test_genes, train_files, test_files)
# add_some_genes_to_val(train_genes, val_genes, train_files, val_files)

# mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/train'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# train_files = [mypath + '/' + f for f in onlyfiles]

# mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/val'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# val_files = [mypath + '/' + f for f in onlyfiles]

# mypath = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/test'
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# test_files = [mypath + '/' + f for f in onlyfiles]

# print("Final Check")
# get_num_common_genes(train_files, val_files, test_files)





