import random
import torch
import numpy as np
device = torch.device("cpu")
from common.utils import HyperParameters
import math

class PrioritizedBuffer(HyperParameters):
    
    def __init__(self,env_params, capacity = int(2**10), alpha=0.6,epsilon=1.,beta=0.5):        
        self.save_hyperparameters()
        self.size = 0
        self.index = 0
        self.buffer = {'observation': np.empty([capacity, env_params['obs_dim']]),
                       'action': np.empty([capacity, env_params['action_dim']]),
                       'reward': np.empty([capacity, 1]),
                       'done': np.empty([capacity, 1]),
                       'new_observation': np.empty([capacity, env_params['obs_dim']])}
        #Stuff for prioritizing
        self.tree = SegmentTree(self.capacity)
        self.max_priority = epsilon

    def store(self,observation,action,reward,done,new_observation):
        '''
        Add an experience
        :param args: 
            observation,action,reward,done,new_observation,priority
        '''
        self.buffer['observation'][self.index] = observation
        self.buffer['action'][self.index] = action
        self.buffer['reward'][self.index] = reward
        self.buffer['done'][self.index] = done
        self.buffer['new_observation'][self.index] = new_observation
        self.tree.update(self.index, self.max_priority ** self.alpha)
        #size and index update
        self.size = min(self.capacity, self.size + 1)
        self.index = (self.index + 1) % self.capacity
    
    def sample(self, batch_size = 32):
        #empty minibatch
        transitions = np.empty([batch_size], dtype=int)
        weights = np.empty([batch_size])
        #max w_i for weighted importance sampling
        min_prob = self.tree.min / self.tree.sum 
        max_weight = (min_prob * self.size) ** (-self.beta)
        
        segment = self.tree.sum / batch_size
        self.beta = np.min([1., self.beta + 0.001])
        for i in range(batch_size):
            #Sampling a transition
            a = segment * i
            b = segment * (i + 1)
            sample = random.uniform(a, b)
            index = self.tree.get(sample)
            transitions[i] = index
            
            #weighted importance sampling
            prob = self.tree.sumtree[index + self.capacity - 1] / self.tree.sum
            weight = ((prob * self.size) ** (-self.beta)) / max_weight
            weights[i] = weight
                
        observations = torch.as_tensor(self.buffer['observation'][transitions], dtype=torch.float32)
        actions = torch.as_tensor(self.buffer['action'][transitions], dtype=torch.float32)
        rewards = torch.as_tensor(self.buffer['reward'][transitions], dtype=torch.float32)
        dones = torch.as_tensor(self.buffer['done'][transitions], dtype=torch.float32)
        new_observations = torch.as_tensor(self.buffer['new_observation'][transitions], dtype=torch.float32)        
        minibatch = [observations, actions, rewards, dones, new_observations]
                
        return minibatch, transitions, weights
    
    def update_priorities(self,indices,priorities):
        #Updating priorities in tree
        for index,priority in zip(indices,priorities):
            self.max_priority = max(self.max_priority,priority)
            priority = self.max_priority ** self.alpha
            self.tree.update(index, priority)
        return
    
    def print(self):
        maxindex = min(self.size, self.capacity)
        print({key: self.buffer[key][range(maxindex)] for key in self.buffer.keys()})

class SegmentTree:
    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity
        self.capacity = (2 * buffer_capacity - 1)
        self.sumtree = np.zeros((self.capacity)) #nodes of the whole tree  
        self.mintree = np.full((self.capacity), float('inf')) #nodes of the whole tree  
        
    def _propagate(self, leaf_index):
        index = leaf_index
        while index > 0: #until i am at the root
            index = math.floor((index - 1)/2) #my parent
            left = 2*index + 1
            right = 2*index + 2
            self.sumtree[index] = self.sumtree[left] + self.sumtree[right]
            self.mintree[index] = min(self.mintree[left],self.mintree[right])
    
    def update(self, transition_index, priority):
        leaf_index = self.buffer2tree(transition_index)
        self.sumtree[leaf_index] = priority
        self.mintree[leaf_index] = priority
        self._propagate(leaf_index)   
        
    def get(self, sample):
        '''
        Given a sample, i.e. a value between 0 and root
        Returns a transition index
        '''
        index = 0
        while index < self.buffer_capacity - 1:
            left = 2*index + 1
            right = 2*index + 2
            if sample <= self.sumtree[left]:
                index = left
            else:
                sample -= self.sumtree[left]
                index = right
         
        return self.tree2buffer(index)
    
    def tree2buffer(self, index):
        return index - self.buffer_capacity + 1
    def buffer2tree(self, index):
        return index + self.buffer_capacity - 1
    
    @property
    def min(self):
        return self.mintree[0]
    @property
    def sum(self):
        return self.sumtree[0]
                
    def __repr__(self):
        return f"SumTree: {self.sumtree} \n MinTree: {self.mintree}"

if __name__ == "__main__":
    tree = SegmentTree(4)
    tree.update(0,1)
    print(tree)
    tree.update(1,1)
    print(tree)

    '''
    [2, 1.3, 0.7, 1, 0.3, 0.5, 0.2]
    batch_size = 5 -> segment = 0.4
    
    1° sampled value = 0.2
    '''
