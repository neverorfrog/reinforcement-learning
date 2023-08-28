import numpy as np
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gymnasium as gym
from gymnasium.spaces.dict import Dict

# Seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def network(sizes, activation, output_activation = nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        activation = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), activation()]
    return nn.Sequential(*layers)  
  
class Actor(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, env_params, hidden_dims=(400,300), activation=nn.ReLU):
        super().__init__()
        # dimensions
        dimensions = [env_params['obs_dim']] + list(hidden_dims) + [env_params['action_dim']]
        self.pi = network(dimensions, activation, nn.Tanh)
        self.action_bound = env_params['action_bound']
        
    def forward(self, obs):    
        # Return output from network scaled to action space limits.
        return self.action_bound * self.pi(obs)
        
        
class Critic(nn.Module):
    """Parametrized Q Network."""

    def __init__(self, env_params, hidden_dims=(400,300), activation=nn.ReLU):
        super().__init__()
        # dimensions
        dimensions = [env_params['obs_dim'] + env_params['action_dim']] + list(hidden_dims) + [1]
        self.q = network(dimensions, activation)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))    