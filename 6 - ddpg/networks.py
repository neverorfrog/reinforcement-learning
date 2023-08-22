import numpy as np
import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gymnasium as gym

def network(sizes, activation, output_activation = nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        activation = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), activation()]
    return nn.Sequential(*layers)  

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])
  
class Actor(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, env: gym.Env, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()
        # dimensions
        obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        dimensions = [obs_dim] + list(hidden_sizes) + [self.act_dim]
        self.pi = network(dimensions, activation, nn.Tanh)
        self.act_limit = env.action_space.high[0]
        
    def forward(self, obs):        
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs) #TODO tensor?
    
    def select_action(self, obs, noise_weight = 0.3):
        with torch.no_grad(): action = self.pi(obs) #TODO: tensor?
        action += noise_weight * np.random.randn(self.act_dim)
        return np.clip(action, -self.act_limit, self.act_limit)
        
        
class Critic(nn.Module):
    """Parametrized Q Network."""

    def __init__(self, env: gym.Env, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()
        # dimensions
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        dimensions = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        self.q = network(dimensions, activation)

    def forward(self, obs, act):
        qvalue = self.q(torch.cat([obs, act], dim=-1))
        return qvalue
        # return torch.squeeze(qvalue, -1) # Critical to ensure q has right shape.