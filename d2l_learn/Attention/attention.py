import torch

y = torch.arange(6).reshape((2, 3))
print(y)
y = y.reshape(-1)
print(y)




exit()
weight = torch.ones((2, 10)) * 0.1
weight = weight.unsqueeze(1)
value = torch.arange(20).reshape((2, 10))
value.unsqueeze_(-1)
print(weight.shape)