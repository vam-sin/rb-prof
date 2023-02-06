# libraries
import pickle as pkl 
import numpy as np 
from multiprocessing import Process, current_process, Manager
from sknetwork.embedding import Spectral
import sys

# np.set_printoptions(threshold=sys.maxsize)

def getLEM(ind, keys_list, data, embeds_list):
    ss = data[keys_list[ind]][4]
    # print(ss)
    len_ss = len(ss)
    adj = np.zeros((len_ss, len_ss))

    # every nt connected to the one right after it
    for j in range(len_ss-1):
        adj[j][j+1] = 1.0
        adj[j+1][j] = 1.0

    # the loops
    stack = []
    for j in range(len_ss):
        if ss[j] == '(':
            stack.append(j)
        elif ss[j] == ')':
            conn_1 = j 
            conn_2 = stack.pop()
            adj[conn_1][conn_2] = 1.0
            adj[conn_2][conn_1] = 1.0
        else:
            pass 

    adj = np.asarray(adj)
    # print(adj)

    embeds = spectral_decomp.fit_transform(adj)
    embeds_list[keys_list[ind]] = embeds
    print("Got LEM ", ind, embeds.shape)
    sys.stdout.flush()

if __name__ == '__main__':
    with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_nonEmpty_wProtSeq_wRNA-SS.pkl', 'rb') as f:
        data = pkl.load(f)
        f.close()

    keys_list = list(data.keys())

    spectral_decomp = Spectral(32)

    manager = Manager()
    embeds_list = manager.dict()

    num_workers = 16

    for i in range(0, len(keys_list), num_workers):
        worker_pool = []
        for x in range(num_workers):
            p = Process(target=getLEM, args=(i+x, keys_list, data, embeds_list))
            p.start()
            worker_pool.append(p)
        for p in worker_pool:
            p.join()
        # break
        
        # print(len(list(embeds_list.keys())))

    a = input("Finished")

    with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/LEM_files/LEM.pkl', 'wb') as f:
        pkl.dump(embeds_list, f)

'''

'''