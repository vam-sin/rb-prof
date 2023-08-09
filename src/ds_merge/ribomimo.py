'''
generates sequence_standard and labels for all files
'''

# libraries
import itertools
import pandas as pd 
import numpy as np 
import pickle as pkl 
from os import listdir
from os.path import isfile, join
import os

liver_number_to_codon = {idx+1:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G', 'N'], repeat=3))}
liver_codon_to_number = {v: k for k, v in liver_number_to_codon.items()}

out_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/mimo_data_standard/'

file_classes = ['Mohammad16', 'Mohammad19-1', 'Subtelny14'] # 5283, 12441, 16842

for file_class in file_classes:
    print("file_class: ", file_class)
    # load the txt file from ribomimo
    file_path = '../../repos/RiboMIMO/data/' + file_class + '.txt'
    with open(file_path) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    gene_names = []

    for i in range(len(lines)):
        print(i, len(lines))
        # make a pickle file for each gene
        if i%3==0:
            if file_class == 'Mohammad16':
                # mohammed16
                gene_names.append(lines[i].replace('>', '').split('|')[1].split(' ')[0])
            elif file_class == 'Mohammad19-1':
                # mohammed19-1
                gene_names.append(lines[i].replace('>', ''))
            elif file_class == 'Subtelny14':
                # subtelny14
                gene_names.append(lines[i].replace('>', '').replace(' ', ''))
            data_dict = {}
            data_dict['gene'] = gene_names[-1]
            out_file_name_ = out_folder + gene_names[-1] + '_' + file_class + '_CTRL_.pkl'
            with open(out_file_name_, 'wb') as f:
                pkl.dump(data_dict, f)
        elif i%3==1:
            out_file_name_ = out_folder + gene_names[-1] + '_' + file_class + '_CTRL_.pkl'
            # open the pickle file and add the sequence
            with open(out_file_name_, 'rb') as f:
                data_dict = pkl.load(f)

            try:
                codon_split_seq = lines[i].split('\t')
                codon_split_seq = [x.replace(' ', '') for x in codon_split_seq]
            except:
                print("except")
                codon_split_seq = lines[i].split(' ')

            for x in range(len(codon_split_seq)):
                codon_split_seq[x] = liver_codon_to_number[codon_split_seq[x]]

            data_dict['sequence_standard'] = codon_split_seq

            # save the pickle file
            with open(out_file_name_, 'wb') as f:
                pkl.dump(data_dict, f)

        else:
            out_file_name_ = out_folder + gene_names[-1] + '_' + file_class + '_CTRL_.pkl'
            counts = lines[i].split('\t')
            counts = [float(x) for x in counts]

            # normalize the counts
            counts = np.array(counts)
            counts = counts/np.sum(counts)
            
            # open the pickle file and add the counts
            with open(out_file_name_, 'rb') as f:
                data_dict = pkl.load(f)
            
            data_dict['y'] = counts

            # save the pickle file
            with open(out_file_name_, 'wb') as f:
                pkl.dump(data_dict, f)

            # check if percentage annotation is greater than 60% 
            num_counts = np.sum(counts>0.0)
            if num_counts/len(counts) < 0.6:
                # delete the file
                cmd = 'rm ' + out_file_name_
                os.system(cmd)
            