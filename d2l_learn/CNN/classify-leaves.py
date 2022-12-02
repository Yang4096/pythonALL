import torch
import torch.nn as nn
import d2l.torch as d2l
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import utils
import os
import shutil
from tqdm import tqdm
from torch.nn import functional as F


class myDataset(Dataset):
    def __init__(self, address):
        self.raw_data = pd.read_csv(address, sep=',' )
        self.raw_data = np.array(self.raw_data)
    
    def __getitem__(self, index):
        address, label = self.raw_data[index]
        imag = cv2.imread(r'./d2l_learn/torch_test/data/Leaves/'+address)
        imag = np.transpose(imag, (2, 0, 1))
              
        return imag.astype(np.float32), label
    
    def __len__(self):
        return len(self.raw_data)
              



def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters()))

    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0]/metric[1]


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train(net, train_iter, test_iter, num_epochs, lr, device, location):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)
    
    net.apply(init_weights)
    
    net.to(device)
    print(f'training on {device}')
    
    writer = SummaryWriter(location)
    num_batchs = len(train_iter)
    timer = d2l.Timer()
    
    for epoch in tqdm(range(num_epochs)):
        metric = d2l.Accumulator(3)
        net.train()
        
        for i,(X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            
            
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            writer.add_scalars('main_tag', {'train_l': train_l, 'train_acc':train_acc},(epoch+i/num_batchs) * 1000)
            timer.stop()
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        writer.add_scalars('main_tag', {'test_acc':test_acc},(epoch) * 1000)
    
    print(f'last epoch: train_l:{train_l:0.3} train_acc:{train_acc:0.3f} test_acc:{test_acc:0.3f}')
    print(f'{metric[0] * num_epochs /timer.sum() : 0.1f} examples/sec, on device: {str(device)}')
            
    return writer
        
    
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
                                                                                                          
    def forward(self, X):
        y = self.conv1(X)
        y = F.relu(self.bn1(y))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            X = self.conv3(X)
        y += X
        return F.relu(y)
       

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=2, stride=2))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
        
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# net = nn.Sequential(b1, b2, b3, b4, b5,
#                     nn.AdaptiveAvgPool2d((1,1)),
#                     nn.Flatten(), nn.Linear(512, 176))

net = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(44944, 4096), nn.BatchNorm1d(4096), nn.Sigmoid(),
    nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.Sigmoid(),
    nn.Linear(1024, 176))


# X = torch.rand(size=(256, 3, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t', X.shape)

# exit()

batch_size =256
        
trainDataset = myDataset(r'./d2l_learn/torch_test/data/Leaves/digitization_labels_data.csv')
train_iter = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=4)

testDataset = myDataset(r'./d2l_learn/torch_test/data/Leaves/digitization_labels_test.csv')
test_iter = DataLoader(trainDataset, batch_size=batch_size, shuffle=False, num_workers=4)


# torch.uint8
# torch.int64

if __name__=='__main__':
    print('strat')
    
    if os.path.exists(r'./d2l_learn/torch_test/path'):
        shutil.rmtree(r'./d2l_learn/torch_test/path')
        os.mkdir(r'./d2l_learn/torch_test/path')
    
    num_epochs = 50
    lr = 0.01
    location = r'./d2l_learn/torch_test/path'
    device = 'cuda:5'
    writer = train(net, train_iter, test_iter, num_epochs, lr, device, location)
        