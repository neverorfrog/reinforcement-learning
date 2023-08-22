from collections import deque
from copy import deepcopy
import numpy as np
import gymnasium as gym
from networks import *
from tools.plotting import ProgressBoard
from tools.agent import *
from tools.utils import *
import torch
import torch.nn as nn
from tools.buffers import UniformBuffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(HyperParameters):
    def __init__(self, name, env: gym.Env, board: ProgressBoard = None, window = 50,
                 tau = 0.1, pi_lr = 0.0005, q_lr = 0.0005, target_update_freq = 10, online_update_freq = 1,
                 batch_size = 5, start_steps = 5000, gamma=0.9, max_steps=200, max_episodes=500, reward_threshold=400):

        # Hyperparameters
        self.save_hyperparameters()

        # Networks
        self.online_actor: Actor = Actor(env).to(device)
        self.target_actor: Actor = deepcopy(self.online_actor).to(device)
        self.online_critic: Critic = Critic(env).to(device)
        self.target_critic: Critic = deepcopy(self.online_critic).to(device)
        self.policy_optimizer = torch.optim.Adam(self.online_actor.parameters(), lr=pi_lr)
        self.value_optimizer = torch.optim.Adam(self.online_critic.parameters(), lr=q_lr)
        self.value_loss_fn = nn.MSELoss()

        # Experience Replay Buffer
        self.memory = UniformBuffer(env)

    def train(self):

        # Life stats
        self.ep = 1
        self.training = True
        self.rewards = deque(maxlen=self.window)
        self.losses = deque(maxlen=self.window)

        # Populating the experience replay memory
        self.populate_buffer()

        while self.training:

            # ep stats
            steps = 0
            self.ep_reward = 0
            self.ep_mean_value_loss = 0.

            # ep termination
            done = False

            # starting point
            observation = torch.FloatTensor(self.env.reset()[0])

            while not done:
                new_observation, done = self.interaction_step(observation)
                
                # Online network update
                if steps % self.online_update_freq == 0:
                    self.learning_step()

                # Copying online network weights into target network
                if steps % self.target_update_freq == 0:
                    self.update_networks()

                # Termination condition satisfied
                if steps > self.max_steps:
                    done = True

                observation = new_observation.detach().clone()
                steps += 1

            self.episode_update()


    def interaction_step(self, observation):
        action = self.online_actor.select_action(observation)
        new_observation, reward, terminated, truncated, _ = self.env.step(action)
        new_observation = torch.FloatTensor(new_observation)
        done = terminated or truncated
        # Storing in the memory
        self.memory.store(observation,action,reward,done,new_observation)    
        # stats
        self.ep_reward += reward
        return new_observation, done
    
    def learning_step(self): 
        # Sampling and loss function
        minibatch = self.memory.sample(batch_size = self.batch_size)
        # Transform batch into torch tensors
        rewards = torch.FloatTensor(minibatch.reward).reshape(-1,1).to(device)
        actions = torch.FloatTensor(minibatch.action).reshape(-1,1).to(device)
        dones = torch.IntTensor(minibatch.done).reshape(-1,1).to(device)
        observations = torch.stack(minibatch.observation).to(device)
        new_observations = torch.stack(minibatch.new_observation).to(device)
        
        #Value Optimization
        estimations = self.online_critic(observations, actions)  
        with torch.no_grad():
            best_actions = self.target_actor(new_observations) #(batch_size, 1)
            target_values = self.target_critic(new_observations, best_actions)
            targets = rewards + (1 - dones) * self.gamma * target_values
        value_loss = self.value_loss_fn(estimations, targets)
        self.ep_mean_value_loss += (1/self.ep)*(value_loss.item() - self.ep_mean_value_loss)        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
                
        #Policy Optimization
        estimated_actions = self.online_actor(observations)
        policy_loss = -self.online_critic(observations, estimated_actions).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def update_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_critic.parameters(), 
                                  self.online_critic.parameters()):
            target_ratio = (1-tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_actor.parameters(), 
                                  self.online_actor.parameters()):
            target_ratio = (1-tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)
        
    def episode_update(self):
        self.rewards.append(self.ep_reward)
        self.losses.append(self.ep_mean_value_loss)
        meanreward = np.mean(self.rewards)
        meanloss = np.mean(self.losses)
        print(f'\rEpisode {self.ep} Mean Reward: {meanreward:.2f} Ep_Reward: {self.ep_reward} Mean Loss: {meanloss:.2f}\t\t')
        # self.board.draw(self.ep, meanreward, self.name)
        if self.ep >= self.max_episodes:
            self.training = False
            print("\nEpisode limit reached")
        if meanreward >= self.reward_threshold:
            self.training = False
            print("\nSUCCESS!")
        self.ep += 1
            
                
    def evaluate(self, env = None, render:bool = True, episodes = 3, max_steps: int = 200):
        mean_reward = 0.
        if env is None: env = self.env
        
        for i in range(1, episodes+1):
            if render: print(f"Starting game {i}")

            observation = torch.FloatTensor(env.reset()[0]) 
            
            terminated = False
            truncated = False
            total_reward = 0
            
            while not terminated and not truncated:
                action = self.online_actor.select_action(observation, noise_weight = 0)
                observation, reward, terminated, truncated, _ = env.step(action)
                observation = torch.FloatTensor(observation)
                total_reward += reward
                if render: self.env.render()
                
            if render: print("\tTotal Reward:", total_reward)
            mean_reward = mean_reward + (1/i)*(total_reward - mean_reward)

        if render: print("Mean Reward: ", mean_reward)
        return mean_reward 
    
    def populate_buffer(self):    
        observation = torch.FloatTensor(self.env.reset()[0])
        for _ in range(self.start_steps):
            with torch.no_grad(): 
                action = self.online_actor.select_action(observation, noise_weight = 1)
            new_observation, reward, terminated, truncated, _ = self.env.step(action)
            new_observation = torch.FloatTensor(new_observation)
            done = terminated or truncated
            self.memory.store(observation,action,reward,done,new_observation)
            observation = new_observation.detach().clone()
            if terminated or truncated: 
                observation = torch.FloatTensor(self.env.reset()[0])
            
    def save(self):
        torch.save(self.network.state_dict(), f"{self.name}.pt")

    def load(self):
        self.network.load_state_dict(torch.load(f"{self.name}.pt"))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret 
         
    
if __name__ == "__main__":
    num_ep = 200
    env = gym.make('InvertedPendulum-v4')
    agent = DDPG("hi", env)
    agent.train()
    
    # testenv = gym.make("MountainCar-v0", render_mode = "human")
    # agent.evaluate(testenv, render = True, episodes = 10, max_steps=200)
