from collections import namedtuple, deque
import random
from torchvision import transforms
import torch
from sumtree import SumTree
import numpy as np


class UniformER:

    def __init__(self, env, n_frames, capacity=50000, burn_in=10000, device=torch.device('cpu')):
        self.capacity = capacity
        self.burn_in = burn_in

        #Deques for storing
        self.transition = namedtuple('transition',('state', 'action', 'reward', 'done', 'next_state'))
        self.transitions = deque(maxlen=capacity)
        self.state = deque(maxlen=n_frames)
        self.next_state = deque(maxlen=n_frames)
        self.n_frames = n_frames

        #Transforms for preprocessing before storing
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize((84,84))
        self.device = device

        #I want to construct it with an already full state (with noop transitions)
        env.reset()
        observation,reward,done,_,_ = env.step(0)
        for i in range(n_frames):
            self.addObservation(observation)
            self.addNextObservation(observation)
        self.store(self.getState(),0,reward,done,self.getNextState())

    def sample_batch(self, batch_size=32):
        transitions = random.sample(self.transitions,batch_size) #that's a list
        batch = self.transition(*zip(*transitions)) #that's a gigantic transition in which every element is actually a list
        return batch

    def preprocessing(self, observation):
        '''
        Input:
            a frame, i.e. a (96,96,3)~(height,width,channels) tensor 
        Output:
            the same frame, but with shape (1,84,84) greyscale and normalized
        '''
        observation = observation.transpose(2,0,1) #Torch wants images in format (channels, height, width)
        observation = torch.from_numpy(observation).float()
        observation = self.rs(observation) # resize
        observation = self.gs(observation) # grayscale
        return (observation/255) # normalize
    
    def addObservation(self,observation):
        self.state.append(self.preprocessing(observation))
    
    def addNextObservation(self,next_observation):
        self.next_state.append(self.preprocessing(next_observation))

    def getState(self):
        state = torch.stack([observation for observation in self.state],0).squeeze()
        return state
    
    def getNextState(self):
        next_state = torch.stack([observation for observation in self.next_state],0).squeeze()
        return next_state

    def store(self,state,action,reward,done,next_state):
        '''
        Add an experience
        :param args: 
            observation(4frame),action,reward,done,next_observation(4frame)
        '''
        self.transitions.append(self.transition(state,action,reward,done,next_state))

    def burn_in_capacity(self):
        return len(self.transitions) / self.burn_in

    def capacity(self):
        return len(self.transitions) / self.memory_size

    def __iter__(self):
       return iter(self.transitions)
    
    def __len__(self):
        return len(self.transitions)
    
    
class PrioritizedER:

    def __init__(self, env, n_frames, alpha = 0.1, epsilon = 0.001,beta = 0.1,capacity=50000,  batch_size = 32):
        self.capacity = capacity

        #Deques for storing
        self.transition = namedtuple('transition',('state', 'action', 'reward', 'done', 'next_state'))
        self.state = deque(maxlen=n_frames)
        self.next_state = deque(maxlen=n_frames)
        self.n_frames = n_frames

        #Transforms for preprocessing before storing
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize((84,84))
        self.device = device
        
        #Stuff for prioritizing
        self.tree = SumTree(self.capacity)
        self.batch_size = batch_size
        self.treeIndices = [0] * self.batch_size #used for updating priorities in tree
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.max_priority = epsilon

        #I want to construct it with an already full state (with noop transitions)
        env.reset()
        observation,reward,done,_,_ = env.step(0)
        for i in range(n_frames):
            self.addObservation(observation)
            self.addNextObservation(observation)
        self.store(self.getState(),0,reward,done,self.getNextState())

    def store(self,state,action,reward,done,next_state,priority=0.05):
        '''
        Add an experience
        :param args: 
            observation(4frame),action,reward,done,next_observation(4frame)
        '''
        priority = self.max_priority
        transition = self.transition(state,action,reward,done,next_state)
        self.tree.add(transition,priority)
    
    def sample_batch(self):
        
        priorities = torch.empty(batch_size, 1, dtype=torch.float)
        transitions = np.zeros(self.batch_size,dtype=object)
        
        samplingRange = self.tree.root / batch_size
        self.beta = np.min([1., self.beta + 0.001])
        
        a,b = 0,0
        for i in range(batch_size):
            a , b = b , b + samplingRange
            sample = random.uniform(a, b)
            (index, priority, transition) = self.tree.get(sample)
            
            self.treeIndices[i] = index
            priorities[i] = priority
            transitions[i] = transition
            
        priorities = torch.FloatTensor(priorities).reshape(-1,1)

        batch = self.transition(*zip(*[transitions[i] for i in range(self.batch_size)])) #batch
        
        #Importance sampling
        probs = priorities / self.tree.root
        weights = (self.capacity * probs) ** -self.beta
        weights = weights / weights.max()
        
        ok = True
        for i in range(self.batch_size):
            if torch.isnan(weights[i]): ok = False
        if (ok == True): self.weights = weights   
        
        return batch, self.weights

    def preprocessing(self, observation):
        '''
        Input:
            a frame, i.e. a (96,96,3)~(height,width,channels) tensor 
        Output:
            the same frame, but with shape (1,84,84) greyscale and normalized
        '''
        observation = observation.transpose(2,0,1) #Torch wants images in format (channels, height, width)
        observation = torch.from_numpy(observation).float()
        observation = self.rs(observation) # resize
        observation = self.gs(observation) # grayscale
        return (observation/255) # normalize
    
    def addObservation(self,observation):
        self.state.append(self.preprocessing(observation))
    
    def addNextObservation(self,next_observation):
        self.next_state.append(self.preprocessing(next_observation))

    def getState(self):
        state = torch.stack([observation for observation in self.state],0).squeeze()
        return state
    
    def getNextState(self):
        next_state = torch.stack([observation for observation in self.next_state],0).squeeze()
        return next_state
    
    def update_priorities(self,errors):
        priorities = (errors + self.epsilon) ** self.alpha
        
        #Updating priorities in treee
        for index,priority in zip(self.treeIndices,priorities):
            self.tree.update(index,priority,isLeaf=True)
            self.max_priority = max(self.max_priority,priority)

        return
  