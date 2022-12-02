import numpy as np
import d2l.torch as d2l
import torch
import torch.nn as nn
import utils

y_hat = torch.Tensor([[0, 1, 0],
                      [0, 1, 0],
                      [1, 0, 0]])

y = torch.Tensor([2, 1, 1])

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

def my_accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.sum())


def evaluate_accuracy(net, data_iter, device=None):
    net.eval()
    with torch.no_grad(): 
        if not device:
            device = next(iter(net.parameters()))
        metric = utils.Accumulator(2)
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(utils.accuracy(net(X), y), y.shape[0])
            print(metric[0] , metric[1])
    return metric[0] / metric[1]
    

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

net.to('cuda:1')
       
print(evaluate_accuracy(net, test_iter))
