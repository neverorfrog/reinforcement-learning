import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

'''
This class takes the number of inputs and outputs and returns a vector of dimension n_outputs, which contains
the q-function (practically one for each action), through the forward function

Practically this is an estimation of the q value through the neural network
'''
class Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, bias=False,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device

        #Convolutional Layers (take as input the 4 frames stacked upon each other)
        # n_frames = 4
        # bias = False
        self.conv1 = nn.Conv2d(n_inputs,32,kernel_size=4,stride=4,bias=bias)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2,bias=bias)

        #Linear layers
        # 32, 64, 9, 9
        self.linear1 = nn.Linear(32*64*9*9,128,bias=bias)
        self.linear2 = nn.Linear(128,256,bias=bias)
        self.linear3 = nn.Linear(256,n_outputs,bias=bias)


    def forward(self, x):
        '''
        Input:
            The last four observed frames (already preprocessed)
        Output:
            An estimation of the q values, one for each action
        '''
        # Convolutions
        # print("Shape of the state before convolution"); print(x.shape) #torch.Size([32, 4, 84, 84]) is this the right shape?
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print("Shape of the state after convolution"); print(x.shape) #orch.Size([32, 64, 9, 9]) is this the right shape?

        # Linear Layers
        x = torch.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y = self.linear3(x)
        # print(y.shape) #5
        return y
