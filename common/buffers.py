from collections import namedtuple, deque
import random
import torch
import numpy as np
device = torch.device("cpu")
import math
from common.utils import HyperParameters

random.seed(42)
np.random.seed(42)

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
    
          

class FinalBuffer:
    def __init__(self, env_params, replay_strategy = 'future', replay_k = 4, reward_function = None, capacity = 1e6):
        
        #General hyper parameters of the buffer
        self.env_params = env_params
        self.capacity = capacity
        
        #Stuff for replaying transitions with a different goal
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_function = reward_function

        #Stuff for keeping track of size
        self.T = env_params['max_steps']
        self.ep_capacity = int(capacity // self.T)
        self.ep_size = 0
        
        #Implementing here a flattened buffer 
        self.buffer = {'observation': np.empty([self.ep_capacity, self.T, env_params['obs_dim']]),
                       'action': np.empty([self.ep_capacity, self.T, env_params['action_dim']]),
                       'reward': np.empty([self.ep_capacity, self.T]),
                       'new_observation': np.empty([self.ep_capacity, self.T, env_params['obs_dim']]),
                       'goal': np.empty([self.ep_capacity, self.T, env_params['goal_dim']]),
        }

    def store(self,episode, store_again = True):
        observations, actions, rewards, new_observations, goals, achieved_goals = episode
        ep_index = self.ep_size % self.ep_capacity
        self.buffer['observation'][ep_index] = observations
        self.buffer['action'][ep_index] = actions
        self.buffer['reward'][ep_index] = rewards
        self.buffer['new_observation'][ep_index] = new_observations
        self.buffer['goal'][ep_index] = goals
        self.ep_size += 1
        if store_again:
            #compute new episode
            #observations, actions and new_observations stay the same
            goals = np.array([achieved_goals[-1]]*self.env_params['max_steps'])
            rewards = np.array([self.reward_function(ag,g,None) for ag,g in zip(achieved_goals,goals)])
            #save it
            self.store([observations, actions, rewards, new_observations, goals, achieved_goals], store_again = False)        

    def sample(self, batch_size=32):
        #select episodes and timesteps to replay
        max_ep_index = min(self.ep_size, self.ep_capacity - 1)
        episodes = np.random.randint(0, max_ep_index, size=batch_size)
        timesteps = np.random.randint(self.T, size=batch_size)
        transitions = {key: self.buffer[key][episodes,timesteps].copy() for key in self.buffer.keys()}
        return transitions
    
    
    
class FutureBuffer:
    def __init__(self, env_params, replay_strategy = 'future', replay_k = 4, reward_function = None, capacity = 1e6):
        
        #General hyper parameters of the buffer
        self.env_params = env_params
        self.capacity = capacity
        
        #Stuff for replaying transitions with a different goal
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_function = reward_function

        #Stuff for keeping track of size
        self.T = env_params['max_steps']
        self.ep_capacity = int(capacity // self.T)
        self.ep_size = 0
        
        #Implementing here a flattened buffer 
        self.buffer = {'observation': np.empty([self.ep_capacity, self.T, env_params['obs_dim']]),
                       'action': np.empty([self.ep_capacity, self.T, env_params['action_dim']]),
                       'reward': np.empty([self.ep_capacity, self.T]),
                       'new_observation': np.empty([self.ep_capacity, self.T, env_params['obs_dim']]),
                       'goal': np.empty([self.ep_capacity, self.T, env_params['goal_dim']]),
                       'achieved_goal': np.empty([self.ep_capacity, self.T, env_params['goal_dim']]),
                       'new_achieved_goal': np.empty([self.ep_capacity, self.T, env_params['goal_dim']]),}

    def store(self,episode):
        observations, actions, desired, achieved, new_observations, new_achieved_goals = episode
        ep_index = self.ep_size % self.ep_capacity
        self.buffer['observation'][ep_index] = observations
        self.buffer['action'][ep_index] = actions
        self.buffer['new_observation'][ep_index] = new_observations
        self.buffer['goal'][ep_index] = desired
        self.buffer['achieved_goal'][ep_index] = achieved
        self.buffer['new_achieved_goal'][ep_index] = new_achieved_goals
        self.ep_size += 1        

    def sample(self, batch_size=32):
        #select episodes and timesteps to replay
        max_ep_index = min(self.ep_size, self.ep_capacity - 1)
        episodes = np.random.randint(0, max_ep_index, size=batch_size)
        timesteps = np.random.randint(self.T, size=batch_size)
        transitions = {key: self.buffer[key][episodes,timesteps].copy() for key in self.buffer.keys()}
        #indices from HER sampling
        indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = (np.random.uniform(size=batch_size) * (self.T - timesteps)).astype(int)
        future_timesteps = (timesteps + future_offset)[indices]
        #replace goal with future achieved goal
        future_goals = self.buffer['achieved_goal'][episodes[indices],future_timesteps]
        transitions['goal'][indices] = future_goals
        #recompute rewards of changed transitions so that the new achieved goal maybe meets the changed goal
        transitions['reward'] = [self.reward_function(ag,g,None) for ag,g in zip(transitions['new_achieved_goal'],transitions['goal'])]       
        return transitions 
    
