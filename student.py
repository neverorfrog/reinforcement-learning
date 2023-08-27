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
        
    def act(self, state):
        
        #epsilon greedy action selection
        with torch.no_grad():
            if random.random() <= self.epsilon:
                return self.env.action_space.sample()
            else:
                return self.network.greedy_action(torch.FloatTensor(state))

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
        self.epsilon = 0.9
        self.gamma = 0.99
        self.n_episodes = 500
        
        #Neural network stuff for training
        self.target_sync_frequency = 50
        self.network_update_frequency = 10
        self.loss_fn = nn.MSELoss()

        for episode in range(self.n_episodes):
            self.observation,_ = self.env.reset()
            done = False
            
            # stats
            steps = 0
            rewards_ep = 0
            losses = []
            episode_rewards = []

            #Main Training Loop
            while not done:
                #Taking a step
                action = self.act(self.observation)
                #Transition goes into the memory
                next_observation, reward, done, _, _ = self.env.step(action)
                self.memory.store(self.observation,action,reward,done,next_observation)
                self.observation = next_observation.copy()
                
                #stats
                rewards_ep += reward
                
                #Network update
                if steps % self.network_update_frequency == 0:
                    self.update() 

                #Target network sync
                if steps % self.target_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

                steps += 1
                
                if done:
                    if self.epsilon > 0.1:
                        self.epsilon = self.epsilon * 0.99
                    episode_rewards.append(rewards_ep)
                    if (episode+1) % 20 == 0:
                        mean_reward = np.mean(episode_rewards)
                        print("\rMean reward {0} at episode {1} with epsilon={2}"
                            .format(mean_reward,episode,self.epsilon),end=" ")
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
    
    
