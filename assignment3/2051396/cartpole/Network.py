import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
This class takes the number of inputs and outputs and returns a vector of dimension n_outputs, which contains
the q-function (practically one for each action), through the forward function

Practically this is an estimation of the q value through the neural network
'''
class Network(nn.Module):
    def __init__(self,n_inputs,n_outputs,bias=True):
        super().__init__()
        #Linear layers
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(4,64,bias=bias)
        self.linear2 = nn.Linear(64,32,bias=bias)
        self.linear3 = nn.Linear(32,2,bias=bias)

    def forward(self, x):
        '''
        Input:
            The last four observed frames (already preprocessed)
        Output:
            An estimation of the q values, one for each action
        '''
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        y = self.linear3(x)
        return y
    
class DQN(nn.Module):
    
    def __init__(self, env,  learning_rate=0.001):
        super(DQN, self).__init__()

        self.network = Network(env.observation_space._shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)

    def Q(self,state):
        out = self.network(state)
        return out
    
    def greedy_action(self, state):
        qvals = self.Q(state)
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a
    

