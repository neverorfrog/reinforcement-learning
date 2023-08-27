from collections import namedtuple, deque
import random
import torch
import numpy as np
device = torch.device("cpu")
import math
from tools.utils import HyperParameters

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
    
          

class HERBuffer:
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
        
        
class PrioritizedBuffer(HyperParameters):
    
    def __init__(self,env_params, capacity = 50000, alpha=0.8,epsilon=1,beta=0.3, batch_size = 32):        
        self.save_hyperparameters()
        self.size = 0
        self.buffer = {'observation': np.empty([capacity, env_params['obs_dim']]),
                       'action': np.empty([capacity, env_params['action_dim']]),
                       'reward': np.empty([capacity, 1]),
                       'done': np.empty([capacity, 1]),
                       'new_observation': np.empty([capacity, env_params['obs_dim']]),
                       'priority': np.empty([capacity, 1])}
        #Stuff for prioritizing
        self.tree = SumTree(self.capacity)
        self.treeIndices = [0] * self.batch_size #used for updating priorities in tree
        self.batchIndices = [0] * self.batch_size #used for updating priorities in memory
        self.max_priority = epsilon

    def store(self,observation,action,reward,done,new_observation):
        '''
        Add an experience
        :param args: 
            observation,action,reward,done,new_observation,priority
        '''
        index = self.size % self.capacity
        self.buffer['observation'][index] = observation
        self.buffer['action'][index] = action
        self.buffer['reward'][index] = reward
        self.buffer['done'][index] = done
        self.buffer['new_observation'][index] = new_observation
        self.size += 1
        self.tree.add(index,self.max_priority ** self.alpha)
    
    def sample(self):
        
        priorities = np.zeros(self.batch_size)
        transitions = []
        
        samplingRange = self.tree.root / self.batch_size
        self.beta = np.min([1., self.beta + 0.001])

        for i in range(self.batch_size):
            a = samplingRange * i
            b = samplingRange * (i + 1)
            sample = random.uniform(a, b)
            
            (index, priority, transition) = self.tree.get(sample)
            
            self.treeIndices[i] = index
            priorities[i] = priority
            transitions.append(int(transition))
            
        observations = torch.as_tensor(self.buffer['observation'][transitions], dtype=torch.float32)
        actions = torch.as_tensor(self.buffer['action'][transitions], dtype=torch.float32)
        rewards = torch.as_tensor(self.buffer['reward'][transitions], dtype=torch.float32)
        dones = torch.as_tensor(self.buffer['done'][transitions], dtype=torch.float32)
        new_observations = torch.as_tensor(self.buffer['new_observation'][transitions], dtype=torch.float32)        
        batch = [observations, actions, rewards, dones, new_observations]
                
        #Importance Sampling
        probs = priorities / self.tree.root
        weights = (self.tree.size * probs) ** -self.beta
        
        self.weights = weights / weights.max()
        
        # ok = True
        # for i in range(self.batch_size):
        #     if torch.isnan(weights[i]): ok = False
        # if (ok == True): self.weights = weights    
        
        print(self.weights)        
        
        return batch, self.weights
    
    def update_priorities(self,errors):
        if isinstance(errors, torch.Tensor):
            errors = errors.detach().cpu().numpy()
        priorities = (errors + self.epsilon) ** self.alpha
        
        #Updating priorities in tree
        for index,priority in zip(self.treeIndices,priorities):
            self.tree.update(index,priority,isLeaf=True)
            self.max_priority = max(self.max_priority,priority)
        return
    
    def __repr__(self):
        return {key: self.buffer[key][range(self.size)] for key in self.buffer.keys()}
    

class SumTree:
    def __init__(self,n_priorities): #n_priorities = capacity of buffer
        self.capacity = (2 * n_priorities - 1)
        self.tree = [0] * (2 * n_priorities - 1) #nodes of the whole tree
        self.priorities = [0] * n_priorities  #for printing
        self.transitions = np.zeros(n_priorities)
        self.head = 0 #next index where i insert an element (treeIndex)
        self.size = 0 #number of nodes in the tree
        
    def get(self,sample):
        '''
        Given a sample, i.e. a value between 0 and root
        Returns index of tree and of batch and the priority related to the sample
        '''
        index = 0
        level = 0
        while (index*2 + 2) < self.size:
            left = 2*index + 1
            right = 2*index + 2
            level += 1
            if sample <= self.tree[left]:
                index = left
            else:
                sample -= self.tree[left]
                index = right

        batchIndex = self.tree2batch(index)
        priority = self.priorities[batchIndex]
        transition = self.transitions[batchIndex]

        return index , priority , transition
    
    def tree2batch(self,index):
        assert(index <= self.size), 'accessing a non existant node'
        level = math.floor(math.log(index+1,2))
        return index - 2**level + 1
    
    def batch2tree(self,index,level):
        assert(index < len(self.priorities))
        return index + 2**level - 1   
        
    def add(self,transition,priority):
        self.tree[self.head] = priority
        batchIndex = self.tree2batch(self.head)
        self.priorities[batchIndex] = priority
        self.transitions[batchIndex] = transition
        
        if self.head % 2 == 0: #I'm adding a priority as right child
            priority += self.tree[self.head - 1]
        
        if self.head > 0:
            parent = math.floor((self.head - 1)/2)
            self.update(parent,priority,isLeaf=False)
        
        self.head  = (self.head + 1) % self.capacity
        self.size  = min(self.size+1, self.capacity)
        return
    
    def update(self,index,priority,isLeaf):
        #Assigning new priority
        change = priority - self.tree[index]
        self.tree[index] = priority
        if isLeaf:
            batchIndex = self.tree2batch(index)
            self.priorities[batchIndex] = priority
        
        #Propagating new priority
        parent = math.floor((index - 1)/2)
        while parent >= 0:
            self.tree[parent] += change
            parent = math.floor((parent - 1)/2)
    
    @property
    def root(self):
        return self.tree[0] #sum of the leaves
    
    def __repr__(self):
        return f"SumTree: \n Nodes={self.tree.__repr__()} \n Priorities={self.priorities.__repr__()} \n Transitions={self.transitions.__repr__()}"
    
# if __name__ == "__main__":
#     tree = SumTree(4)
#     tree.add(10,10)
#     tree.add(5,5)    
#     tree.add(11,11)
#     tree.add(1,1)
    
#     print(tree)
#     print(tree.get(15))
#     tree.update(3,5,True)
#     print(tree)
#     print(tree.tree2batch(5))