# libraries
import pickle as pkl 
import numpy as np
import RNA
import datetime as dt
from multiprocessing import Process, current_process, Manager
import sys

def add_RNA_SS(ind, dict_data, keys_list):
    print('-'*89)
    print(ind, len(keys_list))
    input_seq = dict_data[keys_list[ind]][0]
    struc, mf = RNA.fold(input_seq)
    # print(struc)
    dict_data[keys_list[ind]].append(str(struc))
    #print(dict_data[keys_list[ind]])
    # check_num_with_Add(dict_data, keys_list)
    sys.stdout.flush()

# def check_num_with_Add(dict_data, keys_list):
#     count = 0
#     for m in range(len(keys_list)):
#         if len(dict_data[keys_list[m]]) == 3:
#             count += 1
#     print(count)
#     sys.stdout.flush()

if __name__ == '__main__':

    # import data 
    with open('../../data/rb_prof_Naef/processed_proper/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_wProt.pkl', 'rb') as f:
        dict_seqCounts = pkl.load(f)

    keys_list = list(dict_seqCounts.keys())
    manager = Manager()
    dict_lists = [manager.list(dict_seqCounts[keys_list[x]]) for x in range(len(keys_list))]
    dict_data = manager.dict(zip(keys_list, dict_lists))
    num_workers = 8
    for i in range(0, len(keys_list), num_workers):
        worker_pool = []
        for x in range(num_workers):
            p = Process(target=add_RNA_SS, args=(i+x, dict_data, keys_list))
            p.start()
            worker_pool.append(p)
        for p in worker_pool:
            p.join()  # Wait for all of the workers to finish.
        
        # check_num_with_Add(dict_data, keys_list)
        print(dict_data[keys_list[i]])
        
    a = input("Finished")  # raw_input(...) in Python 2.
    print(len(keys_list))

    with open('../../data/rb_prof_Naef/processed_proper/seq_annot_final/ensembl_Tr_Seq_CTRL_merged_wProt_wRNA_SS.pkl', 'wb') as f:
        pkl.dump(dict_data, f)