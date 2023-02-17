# libraries
import numpy as np 
import torch_geometric
import torch
import pickle as pkl

'''
example:
..((..))..

'''

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_nonEmpty_wProtSeq_wRNA-SS.pkl', 'rb') as f:
    data = pkl.load(f)
    f.close()

# print(data)
keys_list = list(data.keys())
print(keys_list[0])

processed_data = {}

for i in range(len(keys_list)):
    print(i, keys_list[i])
    key_val = keys_list[i]
    fts = data[keys_list[i]]
    fts_dict = dict(zip(['RNA_Seq', 'Counts', 'Gene', 'Protein_Seq', 'RNA_SS'], fts))
    ss = fts_dict['RNA_SS']

    len_ss = len(ss)
    adj = np.zeros((len_ss, len_ss))

    # print(adj)

    # every nt connected to the one right after it
    for j in range(len_ss-1):
        adj[j][j+1] = 1.0
        adj[j+1][j] = 1.0

    # print(adj)

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

    # adj = torch.from_numpy(np.asarray(adj))
    adj = np.asarray(adj)

    # sparse_adj = torch_geometric.utils.dense_to_sparse(adj)

    fts_dict['RNA_SS_Graph'] = adj

    processed_data[key_val] = fts_dict

with open('../../data/rb_prof_Naef/processed_proper/seq_annot_raw/data_CTRL_nonEMpty_wProtSeq_wRNA-SS_wRNA-SS-Graph.pkl', 'wb') as f:
    pkl.dump(processed_data, f)

