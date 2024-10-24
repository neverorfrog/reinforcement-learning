from collections import namedtuple, deque
import random
import torch
import numpy as np
device = torch.device("cpu")
import math


class UniformBuffer:

    def __init__(self, env, capacity=50000, burn_in=10000):
        self.capacity = capacity
        self.burn_in = burn_in

        #Deques for storing
        self.transition = namedtuple('transition',('observation', 'action', 'reward', 'done', 'new_observation'))
        self.buffer = deque(maxlen=capacity)

        #I want to construct it with an already full state (with noop transitions)
        env.reset()
        action = 0
        observation,reward,done,_,_ = env.step(action)
        self.store(observation,action,reward,done,observation)
    
    def store(self,observation,action,reward,done,new_observation):
        '''
        Add an experience
        :param args: observation,action,reward,done,next_observation
        '''
        observation = torch.as_tensor(observation, dtype = torch.float32)
        new_observation = torch.as_tensor(new_observation, dtype = torch.float32)
        self.buffer.append(self.transition(observation,action,reward,done,new_observation))

    def sample(self, batch_size=32):
        transitions = random.sample(self.buffer,batch_size) #that's a list of batch_size transitions
        minibatch = self.transition(*zip(*transitions)) #that's a gigantic transition in which every element is actually a list
        # Transform batch into torch tensors
        observations = torch.stack(minibatch.observation).to(device)
        actions = torch.tensor(minibatch.action, dtype = torch.float32).reshape(-1,1).to(device)
        dones = torch.tensor(minibatch.done, dtype=torch.int).reshape(-1,1).to(device)
        new_observations = torch.stack(minibatch.new_observation).to(device)
        rewards = torch.tensor(minibatch.reward, dtype=torch.float32).reshape(-1,1)
        return observations, actions, rewards, dones, new_observations
    
    def __getitem__(self, key):
        return self.buffer[key]

    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in

    def capacity(self):
        return len(self.buffer) / self.memory_size
    
    def __len__(self):
        return len(self.buffer)