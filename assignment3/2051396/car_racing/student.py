import gym
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

class Policy(nn.Module):

    def __init__(self, device = torch.device("cpu")):
        super(Policy, self).__init__()
        self.device = device

        #Environment
        self.continuous = False
        self.env = gym.make('CarRacing-v2', continuous=self.continuous)
        self.gamma = 0.9
        self.epsilon = 0.3
        self.n_episodes = 500

        #Convolutional Layers (take as input the 4 frames stacked upon each other)
        n_frames = 4
        bias = False
        self.conv1 = nn.Conv2d(n_frames,32,4,bias=bias)# kernel_size=4,stride=4,bias=bias)
        self.conv2 = nn.Conv2d(32, 64,4,bias=bias) #kernel_size=4,stride=2,bias=bias)
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize((64,64))

        #Linear layers
        self.linear1 = nn.Linear(64*7*7,128,bias=bias)
        self.linear2 = nn.Linear(128,256,bias=bias)
        self.linear3 = nn.Linear(256,self.env.action_space.n,bias=bias)

        #Optimizer for gradient descent
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()

        #Experience replay buffer
        self.buffer = ERB()

    def forward(self, x):
        '''
        Input:
            The last four observed frames (already preprocessed)
        Output:
            An estimation of the q values, one for each action
        '''
        # Convolutions
        print(x.shape) #(4,64,64) is this the right shape?
        self = self.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        print(x.shape) #(64,7,7) is this the right shape?

        # Linear Layers
        x = torch.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y = self.linear3(x)
        # print(y.shape) #5
        return y

    def train(self):
        for episode in range(self.n_episodes):
            state = self.env.reset() 
            self.buffer.clear()

            # perform noop for 4 steps
            dummy_observation = np.ndarray((96,96,3))
            dummy_action = 0
            dummy_reward = 0.0
            for i in range(4):
                self.buffer.append(dummy_observation,dummy_action,dummy_reward,False,dummy_observation)
                
            observation,reward,done,_,_ = self.env.step(0)

            done = False
            iteration = 0

            #Main Training Loop
            while not done:
                #Action selection and simulation
                action = self.act(self.buffer.state)
                next_observation, reward, done, _, _ = self.env.step(action)

                #experience goes into the buffer
                self.buffer.append(observation,action,reward,done,next_observation)

                if len(self.buffer) > 32:
                    self.update() #updating the neural network weights
                
                observation = next_observation
        return
    
    def act(self, state):
        #State is actually made of the last 4 frames, so I preprocess and stack them on top of each other
        # print(state[0].shape) #(96,96,3)
        state = torch.vstack([self.preprocessing(observation) for observation in state]).to(self.device)#.unsqueeze(0).to(self.device)
        # print(state.shape) #(4,64,64)

        #epsilon-greedy action selection
        if random.random() > self.epsilon:
            action = self.env.action_space.sample()
        else:
            qvals = self(state)
            action = torch.max(qvals,dim=-1)[1].item() #index of the action corresponding to the max q value
        return action
    
    def preprocessing(self, observation):
        '''
        Input:
            a frame, i.e. a (96,96,3)~(height,width,channels) tensor 
        Output:
            the same frame, but with shape (64x64) and normalized
        '''
        observation = observation[:83,:].transpose(2,0,1) #Torch wants images in format (channels, height, width)
        observation = torch.from_numpy(observation)
        observation = self.gs(observation) # grayscale
        observation = self.rs(observation) # resize
        return observation/255 # normalize
    
    def update(self):
        self.optimizer.zero_grad()
        batch = self.buffer.sample_batch()
        loss = self.calculate_loss(batch)

        #Backpropagation
        loss.backward()
        self.network.optimizer.step()

    def calculate_loss(self,batch):
        #transform in torch tensors
        rewards = torch.FloatTensor(batch.reward)
        actions = torch.LongTensor(batch.action)
        dones = torch.IntTensor(batch.done)
        observations = self.tuple2tensor(batch.observation)
        next_observations = self.tuple2tensor(batch.next_observation)

        print(type(observations))

        #estimated q values
        q_estimated = self.forward(observations)
        estimation = torch.gather(q_estimated, 1, actions)

        #target q values
        with torch.no_grad():
            q_next = self.forward(next_state)
        q_next_max = torch.max(q_next, dim=-1)[0].reshape(-1, 1)
        target = rewards + (1 - dones)*self.gamma*q_next_max
        
        return self.loss_fn(estimation, target)
    
    def tuple2tensor(tuple):
        tensorShape = (len(tuple),*[i for i in tuple[0].shape])
        tensor = torch.zeros(tensorShape)
        for i, x in enumerate(tuple):
            tensor[i] = torch.FloatTensor(x)
        return tensor
        
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt'), map_location=self.device)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret



class ERB:
    def __init__(self, capacity=50000, burn_in=10000):
        self.capacity = capacity
        self.burn_in = burn_in
        self.transition = namedtuple('transition',('observation', 'action', 'reward', 'done', 'next_observation'))
        # self.head = 32
        # self.buffer = np.empty(capacity,dtype=tuple) #using an array for better performance
        self.buffer = deque(maxlen=capacity)
        self.state = deque(maxlen=4)
        self.next_state = deque(maxlen=4)

    def sample_batch(self, batch_size=32):
        # sample_idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        # batch = zip(*[self.buffer[i] for i in sample_idx])
        # sample_idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        # batch = self.transition(*zip([vector[i] for i in sample_idx]))
        transitions = random.sample(self.buffer,batch_size) #that's a list
        batch = self.transition(*zip(*transitions)) #that's a gigantic transition in which every element is actually a list
        return batch

    def append(self, observation,action,reward,done,next_observation):
        '''
        Add an experience
        :param args: observation, action, reward, done, next_observation
        '''
        self.buffer.append(self.transition(observation,action,reward,done,next_observation))
        self.state.append(observation)
        self.next_state.append(next_observation)

    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in

    def capacity(self):
        return len(self.buffer) / self.memory_size

    def __iter__(self):
       ''' Returns the Iterator object '''
       return iter(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        self.state.clear()
        self.next_state.clear()
