import pandas as pd
import numpy as np 

ds = pd.read_csv('/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_proper/merge_gnorm/merge_gnorm_CTRL.csv')

gnorm_lis = list(ds["count_GScale"]) 

print(np.max(gnorm_lis), np.min(gnorm_lis))

'''
Max: 1.0, Min: 7.345325801925945e-06
'''