import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
from d2l import torch as d2l
import torch.utils.tensorboard
import os
import shutil
from tqdm import tqdm
import train_exersise

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


def train(net, train_iter, test_iter, num_epochs, lr, device):
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr)

    net.apply(init_weights)

    print('training on', device)
    net.to(device) 

    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in tqdm(range(num_epochs)):
        # 训练损失值和， 训练准确率之和， 样本数
        metric = d2l.Accumulator(3)
        net.train()

        for i, (X, y) in enumerate(train_iter):
            timer.start()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
             
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches -1:
                writer.add_scalars('inf', {'train_l': train_l, 'train_acc': train_acc}, int(epoch * 1000 + (i + 1) / num_batches * 1000) )
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        writer.add_scalars('inf', {'test_acc': test_acc}, epoch * 1000)
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec', f'on {str(device)}')

    # writer.add_text(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
    #       f'test acc {test_acc:.3f}',0)
    # writer.add_text(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
    #       f'on {str(device)}',1)
batch_norm_net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
    
if __name__=='__main__':
    print(f'start at', os.path.abspath('.'))    
      

    batch_size = 512
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    if os.path.exists('./path/to/log'):# 相对地址有问题！！！！
        shutil.rmtree('./path/to/log')
        os.mkdir('./path/to/log')

    # writer = SummaryWriter('./path/to/log')

    conv_arch = ((1, 64), (1, 128), (1, 256), (2, 512), (2, 512))
    small_conv_arch = [(p[0], int(p[1] / 1)) for p in conv_arch]

    lr, num_epochs = 0.01, 50
    writer = train_exersise.train(batch_norm_net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu(1), './path/to/log')
    # train(batch_norm_net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu(1))
    writer.close()

























