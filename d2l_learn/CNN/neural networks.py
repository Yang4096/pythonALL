import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class dataSet(Dataset):
    def __init__(self, csvFile):
        self.df = np.loadtxt(csvFile, dtype=str, delimiter=',', skiprows=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return torch.from_numpy(self.df[item])


dataset = dataSet('1.csv')
d1 = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, num_workers=0)
idata = iter(d1)
print(next(idata))

