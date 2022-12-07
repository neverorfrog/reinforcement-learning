import gym
import torch
import torch.nn as nn
import numpy as np
import random
from Network import DQN
from ReplayMemory import UniformER
from copy import deepcopy

class Policy(nn.Module):

    def __init__(self, device = torch.device("cpu")):
        super().__init__()
        self.device = device

        #Environment
        self.continuous = False
        self.env = gym.make('CarRacing-v2', continuous=self.continuous)
        self.gamma = 0.9
        self.epsilon = 0.9
        self.n_episodes = 5

        # Neural network
        self.network = DQN(4,self.env.action_space.n)
        self.target_network = deepcopy(self.network)
        #Optimizer for gradient descent
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()

        #Experience replay memory
        self.memory = UniformER(self.env)

    def train(self):
        for episode in range(self.n_episodes):

            self.memory.clear()
            self.memory = UniformER(self.env)
            self.epsilon = 0.9
            
            for j in range(32):
                observation,reward,done,_,_ = self.env.step(1)
                self.memory.store(observation,0,reward,done,observation)

            observation,reward,done,_,_ = self.env.step(0)
            done = False
            steps = 0
            max_steps = 500

            #Main Training Loop
            while not done and steps < max_steps:
                #Action selection and simulation
                action = self.act(self.memory.getState())
                next_observation, reward, done, _, _ = self.env.step(action)

                #experience goes into the memory
                self.memory.store(observation,action,reward,done,next_observation)

                #Network update
                self.update() 
                #Target network update
                if steps % 100:
                    self.target_network.load_state_dict(self.network.state_dict())

                observation = next_observation
                steps = steps + 1
                self.epsilon = self.epsilon * 0.99

        # save models
        self.save_models()
        return
    
    def act(self, state):
        #epsilon-greedy action selection
        state = self.memory.getState()
        # print("State shape {0}".format(state.shape))
        # if (state.shape == (96,96,3)):
        #     self.memory.preprocessing(state)

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
        rewards = torch.FloatTensor(batch.reward).to(self.device).reshape(-1,1)
        actions = torch.LongTensor(batch.action).to(self.device).reshape(-1,1)
        dones = torch.IntTensor(batch.done).to(self.device).reshape(-1,1)
        states = torch.stack(batch.state,0).to(self.device)
        next_states = torch.stack(batch.next_state,0).to(self.device)

        # print(rewards[-1])

        #estimated q values
        q_estimated = self.network(states)
        # print("q"); print(q_estimated.shape) # (32,5)
        # print("actions"); print(actions.shape) #(32,1)
        estimation = torch.gather(q_estimated, 0, actions) #(32,1)

        #target q values
        with torch.no_grad():
            q_estimated_next = self.network(next_states)
            q_target = self.target_network(next_states)

        # print("q_estimated_next"); print(q_estimated_next.shape) #(32,5)
        best_actions = torch.argmax(q_estimated_next,dim=-1).reshape(-1,1)
        # print("best action"); print(best_actions.shape) #(32,1)
        q_target_max = torch.gather(q_target,0,best_actions)
        # print("Q target max = {0}".format(q_target_max.shape)) #(32,1)
        # q_target_max = torch.max(q_next, dim=-1)[0].reshape(-1, 1)

        target = rewards + (1 - dones)*self.gamma*q_target_max
        # print("Target shape = {0}".format(target.shape)) #(32,1)
        
        return self.loss_fn(estimation, target)

    def save_models(self):
        torch.save(self.network, "Q_net")
        
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt'))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
