from collections import deque
from copy import deepcopy
import os
import numpy as np
import gymnasium as gym
from networks import *
import torch
import torch.nn as nn
from buffer import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

'''
Vanilla DDP: 756 episodes 1000 reward 411 mean reward
'''

class DDPG():
    def __init__(self, name, env: gym.Env, window = 50,
                 polyak = 0.995, pi_lr = 0.0001, q_lr = 0.0001, target_update_freq = 1, update_freq = 1, 
                 eps = 1.0, eps_decay = 0.99, batch_size = 64, gamma=0.99, max_episodes=200, reward_threshold=400):

        # Hyperparameters
        self.name = name
        self.env = env
        self.window = window
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq
        self.eps = eps
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.reward_threshold = reward_threshold
        
        # env params for networks and buffer
        observation = env.reset()[0]
        self.env_params = {'obs_dim': observation.shape[0], 
                           'action_dim': env.action_space.shape[0], 
                           'action_bound': env.action_space.high[0],
                           'max_steps': env._max_episode_steps}

        # Networks
        self.actor: Actor = Actor(self.env_params).to(device)
        self.target_actor: Actor = deepcopy(self.actor).to(device)
        self.critic: Critic = Critic(self.env_params).to(device)
        self.target_critic: Critic = deepcopy(self.critic).to(device)
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
        self.value_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)
        self.value_loss_fn = nn.MSELoss()
        #These networks must be updated not through the gradients but with polyak
        for param in self.target_critic.parameters():
            param.requires_grad = False 
        for param in self.target_actor.parameters():
            param.requires_grad = False 

        # Experience Replay Buffer
        self.memory = StandardBuffer(self.env_params)
        self.start_steps = 5*batch_size
        

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
            observation = self.env.reset()[0]

            while not done:
                new_observation, done = self.interaction_step(observation)
                
                # Online network update
                if steps % self.update_freq == 0:
                    self.learning_step()

                # Copying online network weights into target network
                if steps % self.target_update_freq == 0:
                    self.update_target_networks()

                observation = new_observation
                steps += 1

            self.episode_update()


    def interaction_step(self, observation):
        action = self.select_action(observation, noise_weight = self.eps)
        new_observation, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        # Storing in the memory
        self.memory.store(observation,action,reward,done,new_observation) 
        # stats
        self.ep_reward += reward
        return new_observation, done
    
    def select_action(self, obs, noise_weight = 0.5):
        with torch.no_grad(): 
            action = self.actor(torch.as_tensor(obs, dtype=torch.float32))
            action += noise_weight * np.random.randn(self.env_params['action_dim'])
            action = np.clip(action, -self.env_params['action_bound'], self.env_params['action_bound'])
        return action
    
    def learning_step(self): 
        #Sampling of the minibatch
        batch = self.memory.sample(batch_size = self.batch_size)
        observations, actions, rewards, dones, new_observations = batch
        #Value Optimization
        estimations = self.critic(observations, actions)  
        with torch.no_grad():
            best_actions = self.target_actor(new_observations) #(batch_size, 1)
            target_values = self.target_critic(new_observations, best_actions)
            targets = rewards + (1 - dones) * self.gamma * target_values
        value_loss = self.value_loss_fn(estimations, targets)
        self.ep_mean_value_loss += (1/self.ep)*(value_loss.item() - self.ep_mean_value_loss)        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        #Don't waste computational effort
        for param in self.critic.parameters():
            param.requires_grad = False
                
        #Policy Optimization
        estimated_actions = self.actor(observations)
        policy_loss = -self.critic(observations, estimated_actions).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        #Reactivate computational graph for critic
        for param in self.critic.parameters():
            param.requires_grad = True
    
    def update_target_networks(self, polyak=None):
        polyak = self.polyak if polyak is None else polyak
        with torch.no_grad():
            for target, online in zip(self.target_critic.parameters(), 
                                    self.critic.parameters()):
                target.data.mul_(polyak)
                target.data.add_((1 - polyak) * online.data)

            for target, online in zip(self.target_actor.parameters(), 
                                    self.actor.parameters()):
                target.data.mul_(polyak)
                target.data.add_((1 - polyak) * online.data)
        
    def episode_update(self):
        self.eps = max(0.1, self.eps * self.eps_decay)
        self.rewards.append(self.ep_reward)
        self.losses.append(self.ep_mean_value_loss)
        meanreward = np.mean(self.rewards)
        meanloss = np.mean(self.losses)
        print(f'\rEpisode {self.ep} Mean Reward: {meanreward:.2f} Ep_Reward: {self.ep_reward} Mean Loss: {meanloss:.2f}\t\t')
        if self.ep >= self.max_episodes:
            self.training = False
            print("\nEpisode limit reached")
        if meanreward >= self.reward_threshold:
            self.training = False
            print("\nSUCCESS!")
        self.ep += 1
            
                
    def evaluate(self, env = None, render:bool = True, num_ep = 3):
        mean_reward = 0.
        if env is None: env = self.env
        
        for i in range(1, num_ep+1):
            if render: print(f"Starting game {i}")

            observation = torch.FloatTensor(env.reset()[0]) 
            
            terminated = False
            truncated = False
            total_reward = 0
            
            while not terminated and not truncated:
                action = self.select_action(observation, noise_weight = 0)
                observation, reward, terminated, truncated, _ = env.step(action)
                observation = torch.FloatTensor(observation)
                total_reward += reward
                if render: self.env.render()
                
            if render: print("\tTotal Reward:", total_reward)
            mean_reward = mean_reward + (1/i)*(total_reward - mean_reward)

        if render: print("Mean Reward: ", mean_reward)
        return mean_reward 
    
    def populate_buffer(self):    
        observation = self.env.reset()[0]
        for _ in range(self.start_steps):
            with torch.no_grad(): 
                action = self.select_action(observation, noise_weight = 1)
            new_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.memory.store(observation,action,reward,done,new_observation)
            observation = new_observation
            if terminated or truncated: 
                observation = self.env.reset()[0]
            
    def save(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here,"models",self.name)
        
        os.makedirs(path, exist_ok=True)
        
        torch.save(self.actor.state_dict(), open(os.path.join(path,"actor.pt"), "wb"))
        torch.save(self.critic.state_dict(), open(os.path.join(path,"critic.pt"), "wb"))
        print("MODELS SAVED!")

    def load(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "models", self.name)
        
        self.actor.load_state_dict(torch.load(open(os.path.join(path, "actor.pt"), "rb"), weights_only=True))
        self.critic.load_state_dict(torch.load(open(os.path.join(path, "critic.pt"), "rb"), weights_only=True))
        print("MODELS LOADED!")

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret 