import torch
from torch import nn
from d2l import torch as d2l
import torch.nn.functional as F
import math

def sequence_mask(X, valid_lens):
    print(valid_lens.shape)
    mask = torch.arange(0, X.shape[1])[None, :] < valid_lens[:, None]
    X[~mask] = -1e6
    return X


def masked_softomax(X, valid_lens):
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens)
        return F.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self, dropout) -> None:
        super().__init__(self)
        self.dropout = nn.Dropout(dropout)
        
    def forword(self, q, k , v, valid_lens=None):
        d = q.shape[-1]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softomax(scores, valid_lens)
        return torch.bnn(self.dropout(self.attention_weights), v)

