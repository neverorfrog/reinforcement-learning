import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

'''
This class takes the number of inputs and outputs and returns a vector of dimension n_outputs, which contains
the q-function (practically one for each action), through the forward function

Practically this is an estimation of the q value through the neural network
'''
class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs, bias=True,device=torch.device("cpu")):
        super().__init__()
        self.device = device

        #Convolutional Layers (take as input the 4 frames stacked upon each other)
        self.conv1 = nn.Conv2d(n_inputs,32,kernel_size=8,stride=4,bias=bias,device=device)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2,bias=bias, device=device)
        self.conv3 = nn.Conv2d(64,64,kernel_size=2,stride=1,bias=bias, device=device)

        #Linear layers
        self.linear1 = nn.Linear(64*8*8,128,bias=bias,device=device)
        self.linear2 = nn.Linear(128,256,bias=bias,device=device)
        self.linear3 = nn.Linear(256,n_outputs,bias=bias,device=device)


    def forward(self, x):
        '''
        Input:
            The last four observed frames (already preprocessed)
        Output:
            An estimation of the q values, one for each action
        '''
        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Linear Layers
        # print("state shape={0}".format(x.shape)) #(32,64,8,8)
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y = self.linear3(x)
        # print("y"); print(y.shape) # (32,5)
        return y
