import torch.nn as nn
import torch

class Policy(nn.Module):
    
    def __init__(self, in_dim, out_dim, bias = True):
        super().__init__()
        #Linear layers
        self.activation = nn.ReLU()
        self.net = nn.Sequential(nn.Linear(in_dim,32,bias), nn.Tanh(),
                                 nn.Linear(32,out_dim,bias), nn.Softmax(dim = 0))

    
    def forward(self, x) -> torch.Tensor:
        '''
        Input: a state x
        Output: pmf for the actions
        '''
        return self.net(x)