import torch
from torch import nn
from d2l import torch as d2l

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum): #
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        # 如果它是一个全连接层
        if len(X.shape )== 2:
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 1, 2), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 1, 2), keepdim=True)
        y_hat = (X -mean) / torch.sqrt(var)# + eps
        y_hat = gamma*y_hat + beta
        # 记录均值和方差
        moving_mean = momentum*moving_mean + (1 - momentum)*mean
        moving_var = momentum*moving_var + (1 - momentum)*var

    return y_hat, moving_mean, moving_var

class BatchNorm(nn.Module):
    def __init__(self, num_feature, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_feature)
        else:
            shape = (1, num_feature, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.deta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    
    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.deta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9
        )
        return Y

X = torch.randn((2, 1, 28, 28), dtype=torch.float32, device='cuda:1')
# net = BatchNorm(3, 4)
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), 
    nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10)
)
net.to('cuda:1')
Y = net(X)
print(Y.shape)