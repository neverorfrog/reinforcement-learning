import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
This class takes the number of inputs and outputs and returns a vector of dimension n_outputs, which contains
the q-function (practically one for each action), through the forward function

Practically this is an estimation of the q value through the neural network
'''
class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()


    def forward(self, x):
        x = self.activation_function( self.layer1(x) )
        x = self.activation_function( self.layer2(x) )
        y = self.layer3(x)

        return y
