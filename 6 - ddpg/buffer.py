from collections import namedtuple, deque
import random
import torch
import numpy as np
device = torch.device("cpu")
import math
from common.utils import HyperParameters

random.seed(123)
np.random.seed(123)

class StandardBuffer:
    def __init__(self, env_params, capacity = 50000):
        self.env_params = env_params
        self.capacity = capacity
        self.size = 0
        self.buffer = {'observation': np.empty([capacity, env_params['obs_dim']]),
                       'action': np.empty([capacity, env_params['action_dim']]),
                       'reward': np.empty([capacity, 1]),
                       'done': np.empty([capacity, 1]),
                       'new_observation': np.empty([capacity, env_params['obs_dim']])}
        

    def store(self,observation,action,reward,done,new_observation):
        index = self.size % self.capacity
        self.buffer['observation'][index] = observation
        self.buffer['action'][index] = action
        self.buffer['reward'][index] = reward
        self.buffer['done'][index] = done
        self.buffer['new_observation'][index] = new_observation
        self.size += 1
        

    def sample(self, batch_size=32):
        max_batch_index = min(self.size, self.capacity - 1)
        sampled_indices = random.sample(range(max_batch_index), batch_size)
        observations = torch.as_tensor(self.buffer['observation'][sampled_indices], dtype=torch.float32)
        actions = torch.as_tensor(self.buffer['action'][sampled_indices], dtype=torch.float32)
        rewards = torch.as_tensor(self.buffer['reward'][sampled_indices], dtype=torch.float32)
        dones = torch.as_tensor(self.buffer['done'][sampled_indices], dtype=torch.float32)
        new_observations = torch.as_tensor(self.buffer['new_observation'][sampled_indices], dtype=torch.float32)        
        return [observations, actions, rewards, dones, new_observations]
