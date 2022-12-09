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

        #Training Environment
        self.n_frames = 4
        self.continuous = False
        self.env = gym.make('CarRacing-v2', continuous=self.continuous)
        self.gamma = 0.99
        self.epsilon = 0
        self.n_episodes = 50
        
        # Neural network
        self.network = DQN(self.env)
        self.target_network = deepcopy(self.network)
        self.target_sync_frequency = 50
        self.network_update_frequency = 5
        #Optimizer for gradient descent
        self.loss_fn = torch.nn.MSELoss()

        #Experience replay memory
        self.memory = UniformER(self.env,n_frames=self.n_frames) 
        
        
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
                # print("State = {0}".format(state.shape)) #(1,4,84,84)
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
        self.epsilon = 1
        for i in range(100):
            action = self.act(observation)
            next_observation,_,done = self.training_step(action)
            observation = next_observation.copy()
            if done: self.env.reset()
                    
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
                    negative_reward_patience = 100
                else:
                    negative_reward_patience -= 1
                    if negative_reward_patience == 0:
                        done = True
                        reward = -100
                        
                # stats
                rewards_ep += reward
                steps += 1

                #Network update
                #TODO
                #Do i need to put update_frequency to 1, so that
                #the priority of each transition can be stored?
                if steps % self.network_update_frequency == 0:
                    self.update() 
                #Target network update
                if steps % self.target_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if self.epsilon > 0.05:
                self.epsilon = self.epsilon * 0.9
            episode_rewards.append(rewards_ep)
            print("\rReward {0} at episode {1} with epsilon={2}".format(rewards_ep,episode,self.epsilon),end=" ")       
        return        
    
    def update(self):
        #Sampling and loss function
        self.network.optimizer.zero_grad()
        batch = self.memory.sample_batch()
        loss, td_error = self.calculate_loss(batch)

        #Backpropagation
        loss.backward()
        self.network.optimizer.step()

    def calculate_loss(self,batch):
        
        #transform in torch tensors
        rewards = torch.FloatTensor(batch.reward).reshape(-1,1)
        actions = torch.LongTensor(batch.action).reshape(-1,1)
        dones = torch.IntTensor(batch.done).reshape(-1,1)
        states = torch.stack(batch.state,0)
        next_states = torch.stack(batch.next_state,0)

        #Estimate q values
        q_estimated = self.network.Q(states)
        estimation = torch.gather(q_estimated, 1, actions) #(32,1)

        #target q values
        with torch.no_grad():
            q_double = self.network.Q(next_states)
            q_target = self.target_network.Q(next_states)

        #Double DQN
        best_actions = torch.argmax(q_double,1).reshape(-1,1)
        q_target_max = torch.gather(q_target,1,best_actions)
        # print("Q target = {0}".format(q_target)) #(32,1)
        # print("Q target max = {0}".format(q_target_max)) #(32,1)
        # q_target_max = torch.max(q_next, dim=-1)[0].reshape(-1, 1)

        target = rewards + (1 - dones)*self.gamma*q_target_max
        # print("Target shape = {0}".format(target.shape)) #(32,1)
        
        td_error = np.abs(target-estimation)
        
        return self.loss_fn(estimation, target), td_error
        
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt'))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
