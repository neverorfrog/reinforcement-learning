import numpy as np
import gymnasium as gym
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
        self.encoder = encoder
        self.shape = (self.env.action_space.n, encoder.size)
        self.weights = np.random.rand(self.env.action_space.n, encoder.size)
        
    def __call__(self, feats: torch.Tensor):
        '''State is to be considered already featurized'''  
        feats = feats.reshape(-1,1) #column vector
        return self.weights @ feats
    
    def greedy_action(self, state: torch.Tensor):
        q_values = torch.from_numpy(self(self.encoder(state)))
        return torch.argmax(q_values).item()
    