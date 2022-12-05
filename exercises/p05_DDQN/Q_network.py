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



'''
This class encapsulates the neural network and the optimizer and a function which chooses an action and
effectively calculates the q-values
'''
class Q_network(nn.Module):

    def __init__(self, env,  learning_rate=1e-4):
        super(Q_network, self).__init__()

        #TODOnt
        self.network = Net(env.observation_space._shape[0], env.action_space)

        print("Q network:")
        print(self.network)

        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)

    def greedy_action(self, state):
        #TODOnt
        qvals = self.get_qvals(state)
        #index of the action corresponding to the max q value
        greedy_a = torch.max(qvals,dim=-1)[1].item() 
        return greedy_a

    def get_qvals(self, state):
        #TODOnt
        qval = self.network(state)
        return qval
