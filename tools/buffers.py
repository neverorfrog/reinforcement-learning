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
        action = env.action_space.high
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
    
class PrioritizedBuffer:
    
    def __init__(self,env,n_frames,alpha=0.8,epsilon=0.0001,beta=0.3, capacity=50000, batch_size = 32):
        self.capacity = capacity

        #Deques for storing
        self.transition = namedtuple('transition',('state', 'action', 'reward', 'done', 'next_state'))
        self.n_frames = n_frames
        
        #Stuff for prioritizing
        self.tree = SumTree(self.capacity)
        self.batch_size = batch_size
        self.treeIndices = [0] * self.batch_size #used for updating priorities in tree
        self.batchIndices = [0] * self.batch_size #used for updating priorities in memory
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.max_priority = epsilon

        #I want to construct it with an already full state (with noop transitions)
        env.reset()
        observation,reward,done,_,_ = env.step(0)
        self.store(observation,0,reward,done,observation)
        

    def store(self,observation,action,reward,done,next_observation,priority=0):
        '''
        Add an experience
        :param args: 
            observation(1frame),action,reward,done,next_observation(1frame),priority
        '''
        priority = self.max_priority
        observation = torch.FloatTensor(observation)
        next_observation = torch.FloatTensor(next_observation)
        transition = self.transition(observation,action,reward,done,next_observation)
        self.tree.add(transition,priority)
    
    def sample_batch(self):
        
        priorities = np.zeros(self.batch_size)
        transitions = np.zeros(self.batch_size,dtype=object)
        
        samplingRange = self.tree.root / self.batch_size
        self.beta = np.min([1., self.beta + 0.001])

        for i in range(self.batch_size):
            a = samplingRange * i
            b = samplingRange * (i + 1)
            sample = random.uniform(a, b)
            
            (index, priority, transition) = self.tree.get(sample)
            
            self.treeIndices[i] = index
            priorities[i] = priority
            transitions[i] = transition
        
        priorities = torch.FloatTensor(priorities).reshape(-1,1)
        
        print("Priorities: {} ".format(priorities))
            
        batch = self.transition(*zip(*[transitions[i] for i in range(self.batch_size)])) #batch
        
        #Importance Sampling
        probs = priorities / self.tree.root
        # print("Root: {} ".format(self.tree.root))
        # print("Probabilities: {} ".format(probs))
        weights = (self.tree.size * probs) ** -self.beta
        print("Weights: {} ".format(weights))
        # print("Weights max: {} ".format(weights.max()))

        weights = weights / weights.max()
        
        ok = True
        for i in range(self.batch_size):
            if torch.isnan(weights[i]): ok = False
        if (ok == True): self.weights = weights            
        
        return batch, self.weights
    
    def update_priorities(self,errors):
        if isinstance(errors, torch.Tensor):
            errors = errors.detach().cpu().numpy()
        priorities = (errors + self.epsilon) ** self.alpha
        print("Priorities: {} ".format(priorities))
        
        #Updating priorities in tree
        for index,priority in zip(self.treeIndices,priorities):
            self.tree.update(index,priority,isLeaf=True)
            self.max_priority = max(self.max_priority,priority)

        return
    

class SumTree:
    
    def __init__(self,n_priorities):
        self.capacity = (2 * n_priorities - 1)
        self.tree = [0] * (2 * n_priorities - 1) #nodes of the whole tree
        self.priorities = [0] * n_priorities  #for printing
        self.transitions = np.zeros(n_priorities, dtype=object)
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
        level = math.floor(math.log(index+1,2))
        return index - 2**level + 1
    
    def batch2tree(self,index,level):
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
        return f"SumTree(tree={self.tree.__repr__()}, priorities={self.priorities.__repr__()})"
    
        
        
        