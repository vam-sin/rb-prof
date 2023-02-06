# libraries
import pickle as pkl 
import numpy as np
import RNA
import datetime as dt
from multiprocessing import Process, current_process, Manager
import sys
import time

def add_RNA_SS(ind, dict_seqCounts, tr_rna, keys_list):
    print('-'*89)
    print(ind, len(keys_list))
    input_seq = dict_seqCounts[keys_list[ind]][0]
    struc, mf = RNA.fold(input_seq)
    tr_rna[keys_list[ind]] = struc
    sys.stdout.flush()

if __name__ == '__main__':

    # import data 
    with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_nonEmpty_wProtSeq.pkl', 'rb') as f:
        dict_seqCounts = pkl.load(f)
    start = time.time()

    keys_list = list(dict_seqCounts.keys())
    manager = Manager()
    tr_rna = manager.dict()
    num_workers = 48
    for i in range(0, 10000, num_workers):
        worker_pool = []
        for x in range(num_workers):
            p = Process(target=add_RNA_SS, args=(i+x, dict_seqCounts, tr_rna, keys_list))
            p.start()
            worker_pool.append(p)
        for p in worker_pool:
            p.join()

    end = time.time()
    print(end-start)
    sys.stdout.flush()
        
    a = input("Finished") 
    print(len(keys_list))

    with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_RNA-SS_10k.pkl', 'wb') as f:
        pkl.dump(dict(tr_rna), f)