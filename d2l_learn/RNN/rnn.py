import math
from xml.dom import HIERARCHY_REQUEST_ERR
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)    
    
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
 
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    
    return torch.cat(outputs, dim=0), (H,)

class RNNNodelScratch:
    def __init__(self, vocab_size, num_hiddens, device, 
                 get_params, init_state, forward_fn) -> None:
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)




# X = next(iter(train_iter))
# X = X[0]
# state = net.begin_state(X.shape[0], d2l.try_gpu())

# Y, new_state = net(X.to(d2l.try_gpu()), state)

# print(X.shape)
# print(Y.shape, len(new_state), new_state[0].shape)

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum([torch.sum(p.grad ** 2) for p in params]))    
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
    
        
def train(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer() 
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size = X.shape[0], device= device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        print(Y.shape)
        y = Y.T.reshape(-1)
        print(y.shape)
        exit()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 512
net = RNNNodelScratch(len(vocab), num_hiddens, d2l.try_gpu(3), get_params, init_rnn_state, rnn)

num_epochs, lr = 500, 1
train(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(3))














