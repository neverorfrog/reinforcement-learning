import gym
import torch
import torch.nn as nn
import numpy as np
import random
from Network import Network
from ReplayMemory import ReplayMemory

class Policy(nn.Module):

    def __init__(self, device = torch.device("cpu")):
        super().__init__()
        self.device = device

        #Environment
        self.continuous = False
        self.env = gym.make('CarRacing-v2', continuous=self.continuous)
        self.gamma = 0.9
        self.epsilon = 0
        self.n_episodes = 500

        # Neural network
        self.network = Network(4,self.env.action_space.n)
        #Optimizer for gradient descent
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()

        #Experience replay memory
        self.memory = ReplayMemory()

    def train(self):
        for episode in range(self.n_episodes):
            state = self.env.reset() 
            self.memory.clear()

            # perform noop for 4 steps
            observation,reward,done,_,_ = self.env.step(0)
            # observation = self.preprocessing(observation)
            action = 0
            reward = 0.0
            for i in range(4):
                self.memory.update(observation,observation)
            # self.memory.store(state,action,reward,done,state)

            done = False
            iteration = 0

            #Main Training Loop
            while not done:
                #Action selection and simulation
                action = self.act(self.memory.getState)
                next_observation, reward, done, _, _ = self.env.step(action)

                # next_observation = self.preprocessing(next_observation)

                # self.memory.update(observation,next_observation)

                #experience goes into the memory
                self.memory.store(observation,action,reward,done,next_observation)

                if len(self.memory) > 32:
                    self.update() #updating the neural network weights

                observation = next_observation
        return
    
    def act(self, state):
        #State is actually made of the last 4 frames, so I preprocess and stack them on top of each other
        # print(state[0].shape) #(96,96,3)
        # state = torch.vstack([observation for observation in state]).to(self.device)#.unsqueeze(0)
        # state = torch.tensor([observation for observation in self.memory.state])
        # state = torch.vstack([observation for observation in self.memory.state]).to(self.device)
        # print("Preprocessed state"); print(state.shape) #needs to be (64,64,4)?

        #epsilon-greedy action selection
        if random.random() > self.epsilon:
            action = self.env.action_space.sample()
        else:
            qvals = self.network(state)
            action = torch.max(qvals,dim=-1)[1].item() #index of the action corresponding to the max q value
        return action
    
    
    def update(self):
        self.optimizer.zero_grad()
        batch = self.memory.sample_batch()
        loss = self.calculate_loss(batch)

        #Backpropagation
        loss.backward()
        self.optimizer.step()

    def calculate_loss(self,batch):
        #transform in torch tensors
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        dones = torch.IntTensor(batch.done).to(self.device)
        # states = self.tuple2tensor(batch.state).to(self.device)
        # next_states = self.tuple2tensor(batch.next_state).to(self.device)
        states = torch.stack(batch.state,0).to(self.device)
        next_states = torch.stack(batch.next_state,0).to(self.device)

        #estimated q values
        q_estimated = self.network(states)
        # print("q"); print(q_estimated.shape) 5
        estimation = torch.gather(q_estimated, 0, actions).unsqueeze(0)

        #target q values
        with torch.no_grad():
            q_next = self.network(next_states)
        q_next_max = torch.max(q_next, dim=-1)[0].reshape(-1, 1)
        target = rewards + (1 - dones)*self.gamma*q_next_max
        # print(target.shape)
        
        return self.loss_fn(estimation, target)
    
    # def tuple2tensor(self,tuple):
    #     tensorShape = (len(tuple),*[i for i in tuple[0].shape])
    #     tensor = torch.zeros(tensorShape)
    #     for i, x in enumerate(tuple):
    #         tensor[i] = torch.FloatTensor(x)
    #     tensor.unsqueeze(0).squeeze(1)
    #     return tensor
        
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt'), map_location=self.device)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
