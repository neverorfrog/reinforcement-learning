import numpy as np
import gymnasium as gym
from tools.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Estimator:
    '''
    Given a state this class spits out th predicted Q function
    This is done by a linear function (state encoded first) or a neural network
    This base class implements the linear fashion
    '''
    def __init__(self, env: gym.Env, encoder = None):
        self.env = env
        self.shape = (self.env.action_space.n, encoder.size)
        self.weights = np.random.rand(self.env.action_space.n, encoder.size)
        
    def __call__(self, feats):
        '''State is to be considered already featurized'''  
        feats = feats.reshape(-1,1) #column vector
        return self.weights @ feats
    
    
class DQN(nn.Module):
    '''
    This class takes the number of inputs and outputs and returns a vector of dimension n_outputs, which contains
    the q-function (practically one for each action), through the forward function

    Practically this is an estimation of the q value through the neural network
    '''
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        #Linear layers
        self.activation = nn.ReLU()
        self.net = nn.Sequential(nn.Linear(in_dim,64,bias), nn.ReLU(),
                                 nn.Linear(64,32,bias), nn.ReLU(),
                                 nn.Linear(32,out_dim,bias))
        
    
    def forward(self, x):
        '''
        Input:
            The last four observed frames (already preprocessed)
        Output:
            An estimation of the q values, one for each action
        '''
        # a = torch.flatten(x, start_dim = 1, end_dim = -1) #one sample on each row -> X.shape = (m, d)
        return self.net(x)
    

    