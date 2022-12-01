import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        
        self.activation_function= nn.Tanh()

        l1_out = 64 
        self.layer1 = nn.Linear(n_inputs,l1_out,bias=bias)

        l2_out = 32
        self.layer2 = nn.Linear(l1_out,l2_out,bias=bias)

        self.layer3 = nn.Linear(l2_out,n_outputs,bias=bias)


    def forward(self, x):
        x = self.activation_function( self.layer1(x) )
        x = self.activation_function( self.layer2(x) )
        y = self.layer3(x)

        return y

class Q_network(nn.Module):

    def __init__(self, env,  learning_rate=1e-4):
        super(Q_network, self).__init__()

        #TODO
        #self.network = Net( ?? , ??)
        self.network = Net(1, 1)

        print("Q network:")
        print(self.network)

        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)

    def greedy_action(self, state):
        # TODO
        # greedy action = ??
        greedy_a = 0

        return greedy_a

    def get_qvals(self, state):
        #TODO
        #qval = ?
        qval = 0
        return qval
