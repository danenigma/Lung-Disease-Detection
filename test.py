import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt 
import torch

data_dir = 'data/images'
name     = 'val'
table    = pd.read_pickle(os.path.join(data_dir,'{}_table.pkl'.format(name)))
print(len(table))
labels   = [label for label in table.label]
print('abnormal: ', sum(labels)/len(labels))
print('normal: ', 1-(sum(labels)/len(labels)))
prop_normal = 0.635
data_dist = torch.FloatTensor([1-prop_normal, prop_normal])
print(data_dist)
plt.hist(np.array(labels))
plt.show()
