from copy import deepcopy
import os
import numpy as np
import gymnasium as gym
from networks import *
from common.plotting import ProgressBoard
from common.utils import HyperParameters
import torch
import torch.nn as nn
from common.buffers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class FetchAgent(HyperParameters):
    def __init__(self, name, env: gym.Env, board: ProgressBoard = None, window = 50,
                 polyak = 0.95, pi_lr = 0.001, q_lr = 0.001, target_update_freq = 1, update_freq = 1, eps = 0.9, eps_decay = 0.995,
                 num_epochs = 5, batch_size = 64, gamma=0.99, max_steps=200, max_episodes=500, reward_threshold=400):

        # Hyperparameters
        self.save_hyperparameters()

        # env params for networks and buffer
        observation = env.reset()[0]
        self.env_params = {'obs_dim': observation['observation'].shape[0], 
                           'goal_dim': observation['desired_goal'].shape[0], 
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
        self.memory = FutureBuffer(self.env_params, reward_function=env.compute_reward)
        self.start_steps = 5*batch_size
        
        
    def train(self):
        for i in range(1,self.num_epochs+1):
            success = self.train_epoch()
            print(f"EPOCH {i} SUCCESS RATE {np.mean(success)} \n")  

    def train_epoch(self):

        # epoch stats
        self.ep = 1
        self.training = True
        successes = []

        while self.training:

            # empty episode temporary memory
            observations = np.empty([self.env_params['max_steps'], self.env_params['obs_dim']])
            actions = np.empty([self.env_params['max_steps'], self.env_params['action_dim']])
            new_observations = np.empty([self.env_params['max_steps'], self.env_params['obs_dim']])
            achieved_goals = np.empty([self.env_params['max_steps'], self.env_params['goal_dim']])
            desired_goals = np.empty([self.env_params['max_steps'], self.env_params['goal_dim']])
            new_achieved_goals = np.empty([self.env_params['max_steps'], self.env_params['goal_dim']])
            # starting point
            obs_dict = self.env.reset()[0]
            observation = obs_dict['observation']
            achieved = obs_dict['achieved_goal']
            desired = obs_dict['desired_goal'] #never changes in an episode
            success = 0
            
            # Episode playing
            for t in range(self.env_params['max_steps']):
                action = self.select_action(observation,desired,noise_weight = self.eps)
                new_obs_dict, _, _, _, info = self.env.step(action)
                new_observation = new_obs_dict['observation']
                new_achieved = new_obs_dict['achieved_goal']
                #Storing in the temporary memory
                observations[t] = observation
                achieved_goals[t] = achieved
                desired_goals[t] = desired
                actions[t] = action
                new_observations[t] = new_observation
                new_achieved_goals[t] = new_achieved
                #Preparing for next step
                achieved = new_achieved
                observation = new_observation
                if info['is_success']: success = 1 
                   
                                        
            # storing in the memory the entire episode
            self.memory.store([observations, actions, desired, achieved, new_observations, new_achieved_goals])
            # Episode sampling and learning
            for _ in range(20):
                self.learning_step()            
            self.update_target_networks()

            # episode stuff
            self.episode_update()
            successes.append(success)
            
        return successes
            
    
    def select_action(self,obs,goal,noise_weight = 0.5):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype = torch.float32)
            goal = torch.as_tensor(goal, dtype = torch.float32)  
            action = self.actor(obs,goal)
            action += noise_weight * np.random.randn(self.env_params['action_dim'])
            action = np.clip(action, -self.env_params['action_bound'], self.env_params['action_bound'])
        return action.numpy()
    
    def learning_step(self): 
        #Sampling of the minibatch
        transitions = self.memory.sample(batch_size = self.batch_size)
        
        observations = torch.as_tensor(transitions['observation'], dtype=torch.float32)
        goals = torch.as_tensor(transitions['goal'], dtype=torch.float32)
        new_observations = torch.as_tensor(transitions['new_observation'], dtype=torch.float32)
        actions = torch.as_tensor(transitions['action'], dtype=torch.float32)
        rewards = torch.as_tensor(transitions['reward'], dtype=torch.float32).reshape(-1,1)
                
        #Value Optimization
        estimations = self.critic(observations,goals,actions)
        with torch.no_grad():
            best_actions = self.target_actor(new_observations, goals) #(batch_size, 1)
            target_values = self.target_critic(new_observations,goals,best_actions)
            targets = rewards + self.gamma * target_values
        value_loss = self.value_loss_fn(estimations, targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        #Don't waste computational effort
        for param in self.critic.parameters():
            param.requires_grad = False
                
        #Policy Optimization
        estimated_actions = self.actor(observations,goals)
        policy_loss = -self.critic(observations,goals,estimated_actions).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
                
        #Reactivate computational graph for critic
        for param in self.critic.parameters():
            param.requires_grad = True
    
    def update_target_networks(self, polyak=None):
        polyak = self.polyak if polyak is None else polyak
        with torch.no_grad():
            for target, online in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.data.mul_(polyak)
                target.data.add_((1 - polyak) * online.data)

            for target, online in zip(self.target_actor.parameters(), self.actor.parameters()):
                target.data.mul_(polyak)
                target.data.add_((1 - polyak) * online.data)
        
    def episode_update(self):
        self.eps = max(0.3, self.eps * self.eps_decay)
        if self.ep >= self.max_episodes:
            self.training = False
        self.ep += 1
                
    def evaluate(self, num_ep = 5, render = False):
        #Start testing the episodes
        for i in range(1, num_ep+1):
            if render: print(f"Starting game {i}")
            observation = self.env.reset()[0] 
            obs = observation['observation']
            g = observation['desired_goal']
            success = 0            
            for _ in range(self.env_params['max_steps']):
                with torch.no_grad():
                    obs = torch.as_tensor(obs, dtype = torch.float32)
                    goal = torch.as_tensor(g, dtype = torch.float32) 
                    action = self.actor(obs,goal).detach().cpu().numpy()
                observation_new, reward, _, _, info = self.env.step(action)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                if render: self.env.render()
                if info['is_success']: success = 1
            print(f"SUCCESS: {success}")
            
                

            
    def save(self):
        path = os.path.join("models",self.name)
        if not os.path.exists(path): os.mkdir(path)
        torch.save(self.actor.state_dict(), open(os.path.join(path,"actor.pt"), "wb"))
        torch.save(self.critic.state_dict(), open(os.path.join(path,"critic.pt"), "wb"))
        print("MODELS SAVED!")

    def load(self):
        path = os.path.join("models",self.name)
        self.actor.load_state_dict(torch.load(open(os.path.join(path,"actor.pt"),"rb")))
        self.critic.load_state_dict(torch.load(open(os.path.join(path,"critic.pt"),"rb")))
        print("MODELS LOADED!")

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
from enum import Enum
class Mode(Enum):
    TRAIN = 1
    TEST = 2
     
def fetch_reach(num_ep = 500, mode = None):
    if mode == Mode.TRAIN:
        env = gym.make('FetchReach-v2')
        agent = FetchAgent("ddpg_per_fetch_reach_future", env, max_episodes = num_ep, num_epochs=7)
        agent.train()    
        agent.save() #Done training and saving the model
    if mode == Mode.TEST:
        env = gym.make('FetchReach-v2', render_mode = "human")
        agent = FetchAgent("ddpg_per_fetch_reach_future", env)
        agent.load()
        agent.evaluate(num_ep = 10, render = True)
        
def fetch_push(mode = None):
    if mode == Mode.TRAIN:
        env = gym.make('FetchPush-v2')
        agent = FetchAgent("ddpg_per_fetch_push", env, max_episodes = 500, num_epochs=50)
        agent.train()    
        agent.save() #Done training and saving the model
    if mode == Mode.TEST:
        env = gym.make('FetchPush-v2', render_mode = "human")
        agent = FetchAgent("ddpg_per_fetch_push", env)
        agent.load()
        agent.evaluate(num_ep = 10, render = True)
    
          
if __name__ == "__main__":
    fetch_push(Mode.TRAIN)