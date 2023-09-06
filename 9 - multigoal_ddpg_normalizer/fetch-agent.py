from collections import deque
from copy import deepcopy

import pandas as pd
from utils import *
import os
from normalizer import Normalizer
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from networks import *
from plotting import ProgressBoard
import torch
import torch.nn as nn
from buffer import *
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
np.seterr(all="raise")

# Seed
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

'''REACH
VANILLA HER: 229,242  (window 20)
HGR: 180, 174 (window 10) --- 225,212 (window 20)
'''

'''PUSH
VANILLA HER: 
'''

'''PICKANDPLACE

'''
            
class FetchAgent(Parameters):
    def __init__(self, name, env: gym.Env, board: ProgressBoard = None, window = 500, gamma = 0.98, 
                 prioritized = True, polyak = 0.95, pi_lr = 0.001, q_lr = 0.001, eps = 0.3, 
                 noise_eps = 0.2, learning_rate = 2, logging_rate = 1, action_l2 = 1., batch_size = 256, 
                 gradient_steps=50, success_threshold = 0.98, max_episodes=500, norm_clip = 5, obs_clip = 200):

        # Hyperparameters
        self.save_parameters()

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
        T = self.max_episodes * self.env_params['max_steps']
        self.memory = ReplayBuffer(self.env_params, max_timesteps=T, reward_function=env.unwrapped.compute_reward)
        
        # normalizer
        self.o_norm = Normalizer(
            size=self.env_params['obs_dim'], 
            default_clip_range = self.norm_clip
        )
        self.g_norm = Normalizer(
            size=self.env_params['goal_dim'], 
            default_clip_range = self.norm_clip
        )
        
        
    def train(self, plot = False):
        #Life stats
        self.success_rate = []
        self.ep = 0
        self.timestep = 0
        ep_successes = deque(maxlen = self.window)
        successful_windows = 0
        
        #Plotting
        board = ProgressBoard(self.max_episodes, plot_rate=10)
        mean_success = 0
        for self.ep in tqdm(range(1, self.max_episodes)):
            #Storing
            self.store_episode()
            #Learning
            value_loss, policy_loss = 0, 0
            if self.ep % self.learning_rate == 0: #how many episodes are stored before learning
                for _ in range(self.gradient_steps):
                    value_loss, policy_loss = self.learning_step()           
                self.update_target_networks()
                #Logging
                ep_successes.append(self.evaluate())
                if self.ep % self.logging_rate == 0:
                    mean_success = np.mean(ep_successes)
                self.success_rate.append(mean_success)
                print(f"Episode {self.ep+1} SUCCESS RATE {mean_success:.2f} VL {value_loss:.2f} PL {policy_loss:.2f}\n")
                if plot: board.draw(self.ep, mean_success, "success")
                if self.ep > self.window:
                    if mean_success >= self.success_threshold: successful_windows += 1
                if successful_windows >= 10: break
        self.success_rate = np.array(self.success_rate)
          
    def store_episode(self):
        # empty episode temporary memory
        cache = ReplayCache(env_params = self.env_params)
        # starting point
        obs_dict = self.env.reset()[0]
        # Episode playing
        for t in range(self.env_params['max_steps']):
            action = self.select_action(obs_dict['observation'],obs_dict['desired_goal'])
            new_obs_dict, _, _, _, _ = self.env.step(action)
            #Storing in the temporary memory
            cache.store_transition(t, obs_dict, action, new_obs_dict)
            #Preparing for next step
            obs_dict = new_obs_dict
            self.timestep += 1
                                  
        # storing in the memory the entire episode
        self.memory.store(cache)
        self.update_normalizer(cache)
    
    def select_action(self,obs,goal,explore = True):
        with torch.no_grad():
            obs = self.to_torch(self.o_norm.normalize(obs))
            goal = self.to_torch(self.g_norm.normalize(goal))
            action = self.actor(obs, goal).detach().numpy()
            if explore:
                action += self.env_params['action_bound'] * self.noise_eps * np.random.randn(self.env_params['action_dim'])
            action = np.clip(action, -self.env_params['action_bound'], self.env_params['action_bound'])
        return action
    
    def learning_step(self): 
        #Sampling of the minibatch (with preprocessed observations and goals)
        observations, goals, new_observations, actions, rewards, weights = self.sample_batch()
        
        #Value Optimization
        self.critic.train()
        with torch.no_grad():
            best_actions = self.target_actor(new_observations, goals).clamp(-1,1) #(batch_size, 1)
            target_values = self.target_critic(new_observations,goals,best_actions)
            targets = (rewards + self.gamma * target_values).detach()
            clip_return = 1 / (1 - self.gamma)
            targets = torch.clamp(targets, -clip_return, 0).detach()
        estimations = self.critic(observations,goals,actions)
        value_loss = self.value_loss_fn(estimations,targets)
        value_loss = torch.mean(value_loss * weights)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        #For the priorities
        if self.prioritized:
            priorities = torch.abs(targets - estimations).detach().numpy() + 1e-5
            self.memory.update_priorities(priorities)
        
        #Don't waste computational effort
        for param in self.critic.parameters():
            param.requires_grad = False
                
        #Policy Optimization
        self.actor.train()
        estimated_actions = self.actor(observations,goals)
        policy_loss = -self.critic(observations,goals,estimated_actions).mean()
        policy_loss += self.action_l2 * (estimated_actions / self.env_params['action_bound']).pow(2).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
                
        #Reactivate computational graph for critic
        for param in self.critic.parameters():
            param.requires_grad = True
            
        return value_loss, policy_loss
    
    def sample_batch(self):
		# Sample replay buffer 
        if self.prioritized:
            transitions, weights = self.memory.sample(timestep = self.timestep)
            weights = self.to_torch(weights)
        else:
            transitions = self.memory.sample_uniformly()
            weights = 1
           
        #no preprocessing for these 
        actions = self.to_torch(transitions['action'])
        rewards = self.to_torch(transitions['reward']).reshape(-1,1)
        
        # clipping observations
        observations = np.clip(transitions['observation'], -self.obs_clip, self.obs_clip)
        new_observations = np.clip(transitions['new_observation'], -self.obs_clip, self.obs_clip)
        goals = np.clip(transitions['goal'], -self.obs_clip, self.obs_clip)
        
        #normalizing observations
        observations = self.to_torch(self.o_norm.normalize(observations))
        new_observations = self.to_torch(self.o_norm.normalize(new_observations))
        goals = self.to_torch(self.g_norm.normalize(goals))

        return observations, goals, new_observations, actions, rewards, weights
    
    def update_target_networks(self, polyak=None):
        polyak = self.polyak if polyak is None else polyak
        with torch.no_grad():
            for target, online in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.data.mul_(polyak)
                target.data.add_((1 - polyak) * online.data)

            for target, online in zip(self.target_actor.parameters(), self.actor.parameters()):
                target.data.mul_(polyak)
                target.data.add_((1 - polyak) * online.data)
                
    # update the normalizer
    def update_normalizer(self, episode):
        observations, goals = self.memory.sample_episode(episode)
        # pre process the obs and g
        observations = np.clip(observations, -self.obs_clip, self.obs_clip)
        goals = np.clip(goals, -self.obs_clip, self.obs_clip)
        # update
        self.o_norm.update(observations)
        self.g_norm.update(goals)
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
                
    def evaluate(self, render = False):
        # starting point
        obs_dict = self.env.reset()[0]
        observation = obs_dict['observation']
        goal = obs_dict['desired_goal'] #never changes in an episode
        info = None     
        # Episode playing
        for _ in range(self.env_params['max_steps']):
            action = self.select_action(observation,goal,explore = False)
            new_obs_dict, _, _, _, info = self.env.step(action)
            new_observation = new_obs_dict['observation']
            if render: self.env.render()
            #Preparing for next step
            observation = new_observation
        success = 1 if info['is_success'] else 0
        if render: print(f"SUCCESS: {success}")
        return success        
                   
    def save(self):
        path = os.path.join("models",self.name)
        if not os.path.exists(path): os.mkdir(path)
        torch.save(self.actor.state_dict(), open(os.path.join(path,"actor.pt"), "wb"))
        torch.save(self.critic.state_dict(), open(os.path.join(path,"critic.pt"), "wb"))
        torch.save(self.success_rate, open(os.path.join(path,"success.pt"), "wb"))
        print("MODELS SAVED!")

    def load(self):
        path = os.path.join("models",self.name)
        self.actor.load_state_dict(torch.load(open(os.path.join(path,"actor.pt"),"rb")))
        self.critic.load_state_dict(torch.load(open(os.path.join(path,"critic.pt"),"rb")))
        print("MODELS LOADED!")
        
    def plot_success(self):
        path = os.path.join("models",self.name)
        success_rate = torch.load(open(os.path.join(path,"success.pt"),"rb"))
        xaxis = np.arange(start=1,stop=success_rate.size+1) * 100 #* self.env_params['max_steps']
        plt.plot(xaxis, success_rate)
        plt.show(block = True)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    def to_torch(self, array, copy=False):
        if copy:
            return torch.tensor(array, dtype = torch.float32).to(device)
        return torch.as_tensor(array, dtype = torch.float32).to(device)
    
    
SEEDS = [1,2,3]

def meltplot(env_name = 'FetchReach-v2', prioritized = True):
    success_rates = []
    maxlen = 0
    for seed in SEEDS:
        env = gym.make(env_name)
        if prioritized:
            agent = FetchAgent(f"HGR_{env_name}_{seed}", env)
        else:
            agent = FetchAgent(f"HER_{env_name}_{seed}", env)
        path = os.path.join("models",agent.name)
        success_rate = torch.load(open(os.path.join(path,"success.pt"),"rb"))
        if len(success_rate) > maxlen: maxlen = len(success_rate)
        success_rates.append(success_rate.tolist())
    for i in range(len(SEEDS)):
        lastelem = success_rates[i][-1]
        difference = (maxlen - len(success_rates[i]))
        coda = [lastelem]*difference
        success_rates[i].extend(coda)
    data = np.array(success_rates)
    df1 = pd.DataFrame(success_rates).melt()
    df1.rename(columns={"variable": "episode", "value": "mean success"}, inplace=True)
    # sns.set(style="darkgrid", context="talk", palette="rainbow")
    # sns.lineplot(x="episode", y="mean success", data=df1).set(title="HGR")
    # plt.show(block = True)
    
from enum import Enum
class Mode(Enum):
    TRAIN = 1
    TEST = 2
     
def launch(env_name = 'FetchReach-v2', mode = None, prioritized = True):
    if mode == Mode.TRAIN:
        seeds = [0]
        for seed in seeds:
            set_global_seeds(seed)
            env = gym.make(env_name)
            if prioritized:
                agent = FetchAgent(f"HGR_{env_name}_norm", env, max_episodes = 500, window = 100)
            else:
                agent = FetchAgent(f"HER_{env_name}_norm", env, max_episodes = 500, window = 100)
            agent.prioritized = prioritized
            agent.train()    
            agent.save() #Done training and saving the model
            
    if mode == Mode.TEST:
        set_global_seeds(0)
        env = gym.make(env_name, render_mode = "human")
        if prioritized:
            agent = FetchAgent(f"HGR_{env_name}_norm", env)
        else:
            agent = FetchAgent(f"HER_{env_name}_norm", env)
        agent.load()
        agent.plot_success()
        for _ in range(10):
            agent.evaluate(render = True)
            
if __name__ == "__main__":
    reach = 'FetchReach-v2'
    push = 'FetchPush-v2'
    pickandplace = 'FetchPickAndPlace-v2'
    # launch(pickandplace, Mode.TRAIN, prioritized = False)
    # launch(pickandplace, Mode.TRAIN, prioritized = True)
    # launch(push, Mode.TRAIN, prioritized = False)
    # launch(push, Mode.TRAIN, prioritized = True)
    launch(pickandplace, Mode.TRAIN, prioritized = True)
    # launch(reach, Mode.TRAIN, prioritized = True)
    # launch(pickandplace, Mode.TEST, prioritized=True)
    
    