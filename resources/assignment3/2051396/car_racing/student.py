import gym
import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy
from collections import namedtuple, deque
from torchvision import transforms
import math
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):

    def __init__(self, device = torch.device("cpu")):
        super().__init__()
        self.device = device

        #Training Environment
        self.n_frames = 4
        self.continuous = False
        self.env = gym.make('CarRacing-v2', continuous=self.continuous)
        self.gamma = 0.99
        self.epsilon = 0
        self.n_episodes = 10
        
        # Neural network
        self.network = DQN(self.env)
        self.target_network = deepcopy(self.network)
        self.target_sync_frequency = 100
        self.network_update_frequency = 10
        #Optimizer for gradient descent
        self.loss_fn = torch.nn.MSELoss()

        #Experience replay memory
        self.memory = PrioritizedER(self.env,n_frames=self.n_frames) 
        
        
    def act(self, observation):
        #State update (not in the experience replay, but only the state)
        self.memory.addObservation(observation)
        
        #Epsilon-greedy action selection
        with torch.no_grad():
            if random.random() < self.epsilon:
                action = self.env.action_space.sample()
            #In testing phase I always enter the else branch
            else:
                #State extraction
                state = self.memory.getState().unsqueeze(0)
                #Q estimation
                qvals = self.network.Q(state)
                action = torch.argmax(qvals).item() #index of the action corresponding to the max q value
        
        return action
    
    def training_step(self,action):
        next_observation, reward, done, _, _ = self.env.step(action)
        self.memory.addNextObservation(next_observation)
        state = self.memory.getState()
        next_state = self.memory.getNextState()
        self.memory.store(state,action,reward,done,next_state)
        return next_observation,reward,done

    def train(self):
        #stats
        episode_rewards = []
        steps = 0
        
        #Populating the experience replay memory
        observation, _ = self.env.reset()
        self.epsilon = 1 #Epsilon for populating (random)
        for i in range(100):
            action = self.act(observation)
            next_observation,_,done = self.training_step(action)
            observation = next_observation.copy()
            if done: self.env.reset()
        
        self.epsilon = 0.5 #Epsilon for training   
        for episode in range(self.n_episodes):

            #State reset
            observation,_ = self.env.reset()
            self.memory.addObservation(observation)
            next_observation,_,_ = self.training_step(0)
            self.observation = next_observation.copy()
            done = False
            
            # stats
            rewards_ep = 0
            steps = -1
            negative_reward_patience = 50

            #Main Training Loop
            while not done:
                #Taking a step
                action = self.act(self.observation)
                next_observation,reward,done = self.training_step(action)
                self.observation = next_observation.copy()

                # handle patience
                if reward >=0:
                    negative_reward_patience = 50
                else:
                    negative_reward_patience -= 1
                    if negative_reward_patience == 0:
                        done = True
                        reward = -100
                        
                # stats
                rewards_ep += reward
                steps += 1

                #Network update
                if steps % self.network_update_frequency == 0:
                    self.update() 
                #Target network update
                if steps % self.target_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if self.epsilon > 0.2:
                self.epsilon = self.epsilon * 0.99
            episode_rewards.append(rewards_ep)
            print("\rReward {0} at episode {1} with epsilon={2}".format(rewards_ep,episode,self.epsilon),end=" ")       
        return        
    
    def update(self):
        #Sampling and loss function
        self.network.optimizer.zero_grad()
        batch,weights = self.memory.sample_batch()
        loss = self.calculate_loss(batch,weights)

        #Backpropagation
        loss.backward()
        self.network.optimizer.step()

    def calculate_loss(self,batch,weights):
        
        #transform in torch tensors
        rewards = torch.FloatTensor(batch.reward).reshape(-1,1)
        actions = torch.LongTensor(batch.action).reshape(-1,1)
        dones = torch.IntTensor(batch.done).reshape(-1,1)
        states = torch.stack(batch.state,0)
        next_states = torch.stack(batch.next_state,0)

        #Estimate q values
        q_estimated = self.network.Q(states)
        estimations = torch.gather(q_estimated, 1, actions) #(32,1)

        #target q values
        with torch.no_grad():
            q_double = self.network.Q(next_states)
            q_target = self.target_network.Q(next_states)

        #Double DQN
        best_actions = torch.argmax(q_double,1).reshape(-1,1)
        q_target_max = torch.gather(q_target,1,best_actions)
        targets = rewards + (1 - dones)*self.gamma*q_target_max
        
        #Priorities
        errors = targets - estimations
        self.memory.update_priorities(torch.abs(errors).detach())
        #Loss function
        loss = torch.mean(weights * self.loss_fn(estimations, targets))
        
        return loss
        
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt'))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    

class SumTree:
    
    '''
    Class for the prioritized experience replay buffer. Every node is the sum of its two children
    The root contains the sum of all the leaves
    The priorities are stored in the leaves
    We sample based on the sum of all the priorities
    '''
    
    def __init__(self,n_priorities):
        self.capacity = (2 * n_priorities - 1)
        self.tree = [0] * (2 * n_priorities - 1) #nodes of the whole tree
        self.transitions = np.zeros(n_priorities, dtype=object)
        self.head = 0 #next index where i insert an element (treeIndex)
        self.size = 0 #number of nodes in the tree
        
    def get(self,sample):
        '''
        Given a sample, i.e. a value between 0 and root
        Returns index of tree and of batch and the priority related to the sample
        '''
        index = 0
        while (index*2 + 2) < self.size:
            left = 2*index + 1
            right = 2*index + 2
            if sample <= self.tree[left]:
                index = left
            else:
                sample -= self.tree[left]
                index = right

        batchIndex = self.tree2batch(index)
        priority = self.tree[index]
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
    

class PrioritizedER:
    
    '''This class is responsible for sampling a batch and holding the current state (preprocessed)'''

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

    def store(self,state,action,reward,done,next_state,priority=0):
        '''
        Add an experience
        :param args: 
            observation(4frame),action,reward,done,next_observation(4frame)
        '''
        priority = self.max_priority
        transition = self.transition(state,action,reward,done,next_state)
        self.tree.add(transition,priority)
    
    def sample_batch(self):
        
        priorities = torch.empty(self.batch_size, 1, dtype=torch.float)
        transitions = np.zeros(self.batch_size,dtype=object)
        
        samplingRange = self.tree.root / self.batch_size
        self.beta = np.min([1., self.beta + 0.001])
        
        a,b = 0,0
        for i in range(self.batch_size):
            a , b = b , b + samplingRange
            sample = random.uniform(0, self.tree.root)
            (index, priority, transition) = self.tree.get(sample)
            
            self.treeIndices[i] = index
            priorities[i] = priority
            transitions[i] = transition
            
        priorities = torch.FloatTensor(priorities).reshape(-1,1)
        print(f"Priorities ", priorities)

        batch = self.transition(*zip(*[transitions[i] for i in range(self.batch_size)])) #batch
        
        #Importance sampling
        probs = priorities / self.tree.root
        weights = (self.capacity * probs) ** -self.beta
        weights = weights / weights.max()
        
        print(f"Weights ", weights)
        
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
    
    
class Network(nn.Module):
    def __init__(self,n_inputs,n_outputs,bias=True):
        super().__init__()
        #Convolutional Layers (take as input the 4 frames stacked upon each other)
        self.conv1 = nn.Conv2d(n_inputs,32,kernel_size=8,stride=4,bias=bias,device=device)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2,bias=bias, device=device)
        self.conv3 = nn.Conv2d(64,64,kernel_size=2,stride=1,bias=bias, device=device)

        #Linear layers
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(64*8*8,64,bias=bias)
        self.linear2 = nn.Linear(64,32,bias=bias)
        self.linear3 = nn.Linear(32,n_outputs,bias=bias)

    def forward(self, x):
        '''
        Input:
            The last four observed frames (already preprocessed)
        Output:
            An estimation of the q values, one for each action
        '''
        # Convolutions
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        # Linear Layers
        x = torch.flatten(x,start_dim=1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        y = self.linear3(x)
        return y
    
class DQN(nn.Module):
    
    '''
    This class takes the number of inputs and outputs and returns a vector of dimension n_outputs, which contains
    the q-function (practically one for each action), through the forward function

    Practically this is an estimation of the q value through the neural network
    '''

    
    def __init__(self, env,  learning_rate=0.00001):
        super(DQN, self).__init__()

        self.network = Network(4, env.action_space.n)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)

    def Q(self,state):
        out = self.network(state)
        return out
    
    def greedy_action(self, state):
        qvals = self.Q(state)
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a

       
        
        
