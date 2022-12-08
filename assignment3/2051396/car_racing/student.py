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
        self.gamma = 0.9
        self.epsilon = 0.9
        self.n_episodes = 10
        self.max_steps = 200
        self.target_update_frequency = 10
        
        # Neural network
        self.network = DQN(self.n_frames,self.env.action_space.n)
        self.target_network = deepcopy(self.network)
        #Optimizer for gradient descent
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.SmoothL1Loss()

        #Experience replay memory
        self.memory = UniformER(self.env,n_frames=self.n_frames) 

    def train(self):
        #stats
        episode_rewards = []
        
        for episode in range(self.n_episodes):

            self.memory.clear()
            self.memory = UniformER(self.env,n_frames=self.n_frames)
            self.epsilon = 0.3

            observation,reward,done,_,_ = self.env.step(0)
            done = False
            
            # stats
            rewards_ep = 0
            steps = 0

            #Main Training Loop
            while not done and steps < self.max_steps:
                #Action selection and simulation
                # print("ACTING")
                action = self.act(None)
                next_observation, reward, done, _, _ = self.env.step(action)
                
                # stats
                rewards_ep += reward

                #experience goes into the memory
                self.memory.store(observation,action,reward,done,next_observation)

                #Network update
                if len(self.memory) > 32:
                    # print("UPDATING THE WEIGHTS")
                    self.update() 
                #Target network update
                if steps % self.target_update_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

                observation = next_observation
                steps = steps + 1
                # print("STEPS PASSED={}".format(steps))
                self.epsilon = self.epsilon * 0.999
                
            episode_rewards.append(rewards_ep)
            print("Reward {0} at episode {1} in {2} steps".format(rewards_ep, episode, steps))

        # save models
        self.save_models()
        return
    
    def act(self, state):
        #epsilon-greedy action selection
        state = self.memory.getState()
        # print("State shape {0}".format(state.shape))

        if random.random() > self.epsilon:
            action = self.env.action_space.sample()
        else:
            # print("State {0}".format(state.shape)) 
            qvals = self.network(state).cpu()
            # print("qvals={0}".format(qvals))
            action = torch.argmax(qvals).item() #index of the action corresponding to the max q value
            # print("action={0}".format(action))
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

        #estimated q values
        q_estimated = self.network(states).cpu()
        # print("q"); print(q_estimated.shape) # (32,5)
        # print("actions"); print(actions.shape) #(32,1)
        estimation = torch.gather(q_estimated, -1, actions) #(32,1)

        #target q values
        with torch.no_grad():
            q_double = self.network(next_states)
            q_target = self.target_network(next_states)

        # print("q_double {0}".format(q_double)) #(32,5)
        best_actions = torch.argmax(q_double,dim=-1).reshape(-1,1)
        # print("best actions"); print(best_actions) #(32,1)
        q_target_max = torch.gather(q_target,-1,best_actions)
        # print("Q target = {0}".format(q_target)) #(32,1)
        # print("Q target max = {0}".format(q_target_max)) #(32,1)
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
