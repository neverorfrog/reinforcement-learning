import gym
import torch
import torch.nn as nn
import numpy as np
import random
from copy import deepcopy
import math
import torch.nn.functional as F
device = torch.device("cpu")
from collections import namedtuple, deque

class Policy:

    def __init__(self,device=torch.device("cpu")):        
        #Training Environment
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.epsilon = 0
        
        # Neural network
        self.network = DQN(self.env).to(device)
        self.target_network = deepcopy(self.network).to(device)

        #Experience replay memory
        self.memory = PrioritizedER(self.env,n_frames=1)
        
    # def act(self, state):
    #     #epsilon greedy action selection
    #     with torch.no_grad():
    #         if random.random() <= self.epsilon:
    #             return self.env.action_space.sample()
    #         else:
    #             return self.network.greedy_action(torch.FloatTensor(state))

    def train(self):        
        #Populating the experience replay memory
        observation, _ = self.env.reset()
        self.epsilon = 1
        for i in range(1000):
            action = self.act(observation)
            next_observation, reward, done, _, _ = self.env.step(action)
            self.memory.store(observation,action,reward,done,next_observation)
            observation = next_observation.copy()
            if done: self.env.reset()
        
        #Hyperparameters for training
        # self.epsilon = 0.9
        # self.gamma = 0.99
        # self.n_episodes = 500
        
        #Neural network stuff for training
        self.target_sync_frequency = 50
        self.network_update_frequency = 10
        # self.loss_fn = nn.MSELoss()

        for episode in range(self.n_episodes):
            self.observation,_ = self.env.reset()
            done = False
            
            # stats
            # steps = 0
            # rewards_ep = 0
            # losses = []
            # episode_rewards = []
        
            #Main Training Loop
            while not done:
                #Taking a step
                # action = self.act(self.observation)
                #Transition goes into the memory
                # next_observation, reward, done, _, _ = self.env.step(action)
                self.memory.store(self.observation,action,reward,done,next_observation)
                self.observation = next_observation.copy()
                
                #stats
                # rewards_ep += reward
                
                #Network update
                if steps % self.network_update_frequency == 0:
                    self.update() 

                #Target network sync
                if steps % self.target_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

                # steps += 1
                
                # if done:
                #     if self.epsilon > 0.1:
                #         self.epsilon = self.epsilon * 0.99
                #     episode_rewards.append(rewards_ep)
                #     if (episode+1) % 20 == 0:
                #         mean_reward = np.mean(episode_rewards)
                #         print("\rMean reward {0} at episode {1} with epsilon={2}"
                #             .format(mean_reward,episode,self.epsilon),end=" ")
        return
    
    def update(self):
        #Sampling and loss function
        batch,weights = self.memory.sample_batch()
        loss = self.calculate_loss(batch,weights)

        #Backpropagation
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

    def calculate_loss(self,batch,weights):  
        
        if weights is None:
            weights = torch.ones_like(Q) 
          
        #Transform batch into torch tensors
        rewards = torch.FloatTensor(batch.reward).reshape(-1,1).to(device)
        actions = torch.LongTensor(batch.action).reshape(-1,1).to(device)
        dones = torch.IntTensor(batch.done).reshape(-1,1).to(device)
        states = torch.stack(batch.state).to(device)
        next_states = torch.stack(batch.next_state).to(device)
        
        #Estimate q values
        q_estimated = self.network.Q(states)
        print(f"Q in student: ", q_estimated)
        estimations = torch.gather(q_estimated, 1, actions) #(32,1)

        #Target q values
        with torch.no_grad():
            q_double = self.network.Q(next_states)
            q_target = self.target_network.Q(next_states)
        
        #Double DQN
        best_actions = torch.argmax(q_double,1).reshape(-1,1)
        q_target_max = torch.gather(q_target,1,best_actions).detach()
        targets = (rewards + (1 - dones) * self.gamma * q_target_max).detach()
        
        #For the priorities
        errors = torch.abs(targets - estimations).detach()
        
        # print(f"Errors in student: ", errors)
        self.memory.update_priorities(errors)  
              
        #loss function
        loss = torch.mean(self.loss_fn(estimations,targets) * weights)
        # loss = self.loss_fn(estimations,targets)

        return loss
        
    def save(self):
        torch.save(self.network.state_dict(), 'model_cartpole.pt')

    def load(self):
        self.network.load_state_dict(torch.load('model_cartpole.pt'))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    
class PrioritizedER:
    
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
        
        #Updating priorities in treee
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



class Network(nn.Module):
    def __init__(self,n_inputs,n_outputs,bias=True):
        super().__init__()
        #Linear layers
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(4,64,bias=bias)
        self.linear2 = nn.Linear(64,32,bias=bias)
        self.linear3 = nn.Linear(32,2,bias=bias)

    def forward(self, x):
        '''
        Input:
            The last four observed frames (already preprocessed)
        Output:
            An estimation of the q values, one for each action
        '''
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
    
    def __init__(self, env,  learning_rate=0.0001):
        super(DQN, self).__init__()

        self.network = Network(env.observation_space._shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=learning_rate)

    def Q(self,state):
        out = self.network(state)
        return out
    
    def greedy_action(self, state):
        qvals = self.Q(state)
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a
    
    
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
        :param args: observation(1frame),action,reward,done,next_observation(1frame)
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
    

