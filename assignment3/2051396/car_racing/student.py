import gym
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

class Policy(nn.Module):

    def __init__(self, device=torch.device('cpu')):
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
        self.conv1 = nn.Conv2d(in_channels=n_frames, out_channels=32, kernel_size=4,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,stride=2)
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize((84,84))

        #Linear layers
        bias = True
        self.linear1 = nn.Linear(84*84*4,128,bias=bias)
        self.linear2 = nn.Linear(128,256,bias=bias)
        self.linear3 = nn.Linear(256,self.env.action_space.n,bias=bias)

        #Optimizer for gradient descent
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()

        #Experience replay buffer
        self.buffer = ERB()

    def preproc_state(self, state):
        '''
        Input:
            state (4 frames of shape (96x96x3))
        Output:
            state (a )
        '''
        # State Preprocessing
        print(state.shape)
        state = state[:83,:].transpose(2,0,1) #Torch wants images in format (channels, height, width)
        state = torch.from_numpy(state)
        state = self.gs(state) # grayscale
        state = self.rs(state) # resize
        return state/255 # normalize

    def forward(self, x):
        '''
        Input:
            The last four observed frames (already preprocessed)
        Output:
            An estimation of the q values, one for each action
        '''
        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Linear Layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        y = self.layer3(x)
        return y

    
    def act(self, state):
        print(state.shape)
        #State is actually made of the last 4 frames, so I preprocess and stack them on top of each other
        state = torch.vstack([self.preproc_state(s) for s in state]).unsqueeze(0).to(self.device)
        print(state.shape)

        #epsilon-greedy action selection
        if random.random() > self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.greedy_action(state)
        return action
    
    def greedy_action(self, state):
        qvals = self(state)
        greedy_a = torch.max(qvals,dim=-1)[1].item() #index of the action corresponding to the max q value
        return greedy_a

    def train(self):
        for episode in range(self.n_episodes):
            state = env.reset() # state reset
        
        # perform noop for 60 steps (noisy start)
        for i in range(60):
            s,_,_,_,_ = env.step(0)
            self.buffer.append(s,_,_,_,_)
        
        done = False

        

        #Training loop
        while not done:
            #State extraction

            #Action selection and simulation
            action = self.act(s)
            s_next, r, done, _, _ = self.env.step(action)

            #experience in the buffer
            self.buffer.append(s,action,r,done,s_next)
            # total_reward += reward

            if len(self.buffer) > 32:
                self.update() #updating the neural network weights
            
            s = s_next

        return
    
    def update(self):
        self.optimizer.zero_grad()
        batch = self.buffer.sample_batch()
        loss = self.calculate_loss(batch)

        #Backpropagation
        loss.backward()
        self.network.optimizer.step()

    def calculate_loss(self,batch):
        #extract info from batch
        _, actions, rewards, dones, _ = list(batch)

        #transform in torch tensors
        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1)
        dones = torch.IntTensor(dones).reshape(-1, 1)

        #estimated q values
        q_estimated = self.forward(states)
        estimation = torch.gather(q_estimated, 1, actions)

        #target q values
        with torch.no_grad():
            q_next = self.forward(next_states)
        q_next_max = torch.max(q_next, dim=-1)[0].reshape(-1, 1)
        target = rewards + (1 - dones)*self.gamma*q_next_max
        
        return self.loss_fn(estimation, target)
    
    def tuple2tensor(tuple_of_np):
        tensor = torch.zeros((len(tuple_of_np), tuple_of_np[0].shape[0]))
        for i, x in enumerate(tuple_of_np):
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
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.experience = namedtuple('experience',field_names=['observation', 'action', 'reward', 'done', 'next_observation'])
        self.buffer = deque(maxlen=memory_size)
        self.state = deque(maxlen=4)
        self.next_state = deque(maxlen=4)

    def sample_batch(self, batch_size=32):
        sample_idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.buffer[i] for i in sample_idx])
        return batch

    def append(self, obs_0, a, r, d, obs_1):
        self.buffer.append(self.experience(obs_0, a, r, d, obs_1))
        self.state.append(obs_0)
        self.next_state.append(obs_1)

    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in

    def capacity(self):
        return len(self.buffer) / self.memory_size

    def __iter__(self):
       ''' Returns the Iterator object '''
       return iter(self.buffer)
    
    def __len__(self):
        return len(self.buffer)


''' Things to check
-Is the space action discrete? If not, how is it?
'''
