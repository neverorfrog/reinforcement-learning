from collections import namedtuple, deque
import random
import torch
import numpy as np
from SumTree import SumTree


class UniformER:

    def __init__(self, env, n_frames, capacity=50000, burn_in=10000):
        self.capacity = capacity
        self.burn_in = burn_in

        #Deques for storing
        self.transition = namedtuple('transition',('state', 'action', 'reward', 'done', 'next_state'))
        self.transitions = deque(maxlen=capacity)
        self.n_frames = n_frames

        #I want to construct it with an already full state (with noop transitions)
        env.reset()
        observation,reward,done,_,_ = env.step(0)
        self.store(observation,0,reward,done,observation)
    
    def store(self,observation,action,reward,done,next_observation):
        '''
        Add an experience
        :param args: 
            observation(1frame),action,reward,done,next_observation(1frame)
        '''
        observation = torch.FloatTensor(observation)
        next_observation = torch.FloatTensor(next_observation)
        self.transitions.append(self.transition(observation,action,reward,done,next_observation))

    def sample_batch(self, batch_size=32):
        transitions = random.sample(self.transitions,batch_size) #that's a list
        batch = self.transition(*zip(*transitions)) #that's a gigantic transition in which every element is actually a list
        return batch
    
    def getState(self):
        state = self.transitions[-1].next_state #last inserted element
        return state

    def burn_in_capacity(self):
        return len(self.transitions) / self.burn_in

    def capacity(self):
        return len(self.transitions) / self.memory_size
    
    def __len__(self):
        return len(self.transitions)
    
    
class PrioritizedER(UniformER):
    
    def __init__(self,env,n_frames,alpha=2,epsilon=0.5,capacity=50000, burn_in=10000):
        self.capacity = capacity
        self.burn_in = burn_in

        #Deques for storing
        self.transition = namedtuple('transition',('state', 'action', 'reward', 'done', 'next_state'))
        self.transitions = deque(maxlen=capacity)
        self.n_frames = n_frames
        
        self.tree = SumTree(self.capacity)
        self.priorities = deque(maxlen = self.capacity)
        self.batch_size = 32
        self.treeIndices = [0] * self.batch_size #used for updating priorities in tree
        self.batchIndices = [0] * self.batch_size #used for updating priorities in memory
        self.epsilon = epsilon
        self.alpha = alpha

        #I want to construct it with an already full state (with noop transitions)
        env.reset()
        observation,reward,done,_,_ = env.step(0)
        self.store(observation,0,reward,done,observation)
        

    def store(self,observation,action,reward,done,next_observation,priority=10000):
        '''
        Add an experience
        :param args: 
            observation(1frame),action,reward,done,next_observation(1frame),priority
        '''
        observation = torch.FloatTensor(observation)
        next_observation = torch.FloatTensor(next_observation)
        self.transitions.append(self.transition(observation,action,reward,done,next_observation))
        self.priorities.append(priority)
        self.tree.add(priority)
    
    def sample_batch(self,batch_size = 32):
        
        for i in range(batch_size):
            sample = random.uniform(0, self.tree.root())
            self.treeIndices[i], priority, self.batchIndices[i] = self.tree.get(sample)

        return self.transition(*zip(*[self.transitions[i] for i in self.batchIndices])) #batch
    
    def update_priorities(self,errors):
        priorities = self.getPriority(errors)
        
        #Updating priorities in buffer
        for index,priority in zip(self.batchIndices, priorities):
            self.priorities[index] = priority
        #Updating priorities in treee
        for index,priority in zip(self.treeIndices,priorities):
            self.tree.update(index,priority)

        return
    
    def getPriority(self,error):
        return np.power(error.detach() + self.epsilon, self.alpha)
