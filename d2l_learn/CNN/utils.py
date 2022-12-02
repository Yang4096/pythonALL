import torch
import torch.nn as nn
import d2l.torch as d2l
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time



class Accumulator:
    def __init__(self, num):
        self.data = [0.0] * num
    
    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    
class Timer:
    def __init__(self):
        self.times = []
    
    def strat(self):
        self.tic  = time.time()
    
    def stop(self):
        self.times.append(time.time() - self.tic)
        return self.times[-1]
    
    def avg(self):
        return sum(self.times) / len(self.times)
    
    def sum(self):
        return sum(self.times)
    
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
        
        
def accuracy(y_hat, y):
    '''
        在分类任务中使用，
        用于查看 标签和预测值之间的相同的值有几个
    '''
    if len(y_hat.shape) > 1 and y_hat.shape[1] >1:
         y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return sum(cmp)