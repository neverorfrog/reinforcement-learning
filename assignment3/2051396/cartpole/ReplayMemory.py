from collections import namedtuple, deque
import random
from torchvision import transforms
import torch
import numpy as np


class UniformER:

    def __init__(self, env, n_frames, capacity=50000, burn_in=10000):
        self.capacity = capacity
        self.burn_in = burn_in

        #Deques for storing
        self.transition = namedtuple('transition',('state', 'action', 'reward', 'done', 'next_state'))
        self.buffer = deque(maxlen=capacity)
        self.n_frames = n_frames

        #I want to construct it with an already full state (with noop transitions)
        env.reset()
        observation,reward,done,_,_ = env.step(0)
        self.store(observation,0,reward,done,observation)

    def sample_batch(self, batch_size=32):
        transitions = random.sample(self.buffer,batch_size) #that's a list
        batch = self.transition(*zip(*transitions)) #that's a gigantic transition in which every element is actually a list
        return batch
    
    def getState(self):
        state = self.buffer[-1].next_state #last inserted element
        # print("State {}".format(state.shape)) 
        return state

    def store(self,observation,action,reward,done,next_observation):
        '''
        Add an experience
        :param args: 
            observation(1frame),action,reward,done,next_observation(1frame)
        '''
        observation = torch.FloatTensor(observation)
        next_observation = torch.FloatTensor(next_observation)
        self.buffer.append(self.transition(observation,action,reward,done,next_observation))

    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in

    def capacity(self):
        return len(self.buffer) / self.memory_size

    def __iter__(self):
       ''' Returns the Iterator object '''
       return iter(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    