import torch
import torch.nn as nn
import d2l.torch as d2l
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np

labels = []

raw_data = pd.read_csv(r'./d2l_learn/torch_test/data/Leaves/sample_submission.csv', sep=',' )
all_labels = pd.read_csv(r'./d2l_learn/torch_test/data/Leaves/labels.csv', sep=',' )
raw_data = np.array(raw_data)
all_labels = np.array(all_labels)

digitization_labels_data = np.array(raw_data)
for i, example in enumerate(raw_data):
    for j, label in all_labels:
        if label == example[1]:
            digitization_labels_data[i, 1] = j
            
    
dataframe = pd.DataFrame({'image':digitization_labels_data[:, 0], 'labels':digitization_labels_data[:, 1]})
dataframe.to_csv('./d2l_learn/torch_test/data/Leaves/digitization_labels_test.csv', index=False)


exit()