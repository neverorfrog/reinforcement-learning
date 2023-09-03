from collections import deque
from copy import deepcopy
import inspect
import os
from normalizer import Normalizer
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from networks import *
from common.plotting import ProgressBoard
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

class Parameters:
    def save_parameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
            
class FetchAgent(Parameters):
    def __init__(self, name, env: gym.Env, board: ProgressBoard = None, window = 50, gamma = 0.98, clip_obs = 200,
                 polyak = 0.95, pi_lr = 0.001, q_lr = 0.001, eps = 0.3, noise_eps = 0.2, learning_rate = 1,
                 action_l2 = 1., batch_size = 256, num_batches=50, success_threshold = 0.98, max_episodes=500):

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
        self.memory = ReplayBuffer(self.env_params, max_timesteps=self.max_episodes * self.env_params['max_steps'], reward_function=env.unwrapped.compute_reward)
        self.cache = ReplayCache(env_params = self.env_params)
        
        # normalizer
        self.o_norm = Normalizer(
            size=self.env_params['obs_dim'], 
            default_clip_range = 5,
            eps=0.01
        )
        self.g_norm = Normalizer(
            size=self.env_params['goal_dim'],
            default_clip_range = 5,
            eps=0.01
        )
        
        
    def train(self, plot = False):
        #Life stats
        self.success_rate = []
        self.ep = 0
        self.timestep = 0
        ep_successes = deque(maxlen = self.window)
        
        #Plotting
        board = ProgressBoard(self.max_episodes, plot_rate=10)
        
        for self.ep in tqdm(range(self.max_episodes)):
            #Storing
            self.store_episode()
            #Learning
            if self.ep % self.learning_rate == 0:
                for _ in range(self.num_batches):
                    value_loss, policy_loss = self.learning_step()           
                self.update_target_networks()
            #Logging
            ep_successes.append(self.evaluate())
            if self.ep > self.window:
                mean_success = np.mean(ep_successes)
                self.success_rate.append(mean_success)
                print(f"Episode {self.ep+1} SUCCESS RATE {mean_success:.2f} VL {value_loss:.2f} PL {policy_loss:.2f}\n")
                if plot: board.draw(self.ep, mean_success, "success")
                if mean_success >= self.success_threshold: break
        self.success_rate = np.array(self.success_rate)
          
    def store_episode(self):
        # empty episode temporary memory
        # self.cache.reset()
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
        
        # Episode playing
        for t in range(self.env_params['max_steps']):
            self.timestep += 1 
            action = self.select_action(observation,desired)
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
                                  
        # storing in the memory the entire episode
        self.memory.store([observations, actions, desired, achieved, new_observations, new_achieved_goals])
        # # starting point
        # obs_dict = self.env.reset()[0]
        # # Episode playing
        # for _ in range(self.env_params['max_steps']):
        #     action = self.select_action(obs_dict['observation'],obs_dict['desired_goal'])
        #     new_obs_dict, _, _, _, _ = self.env.step(action)
        #     #Storing in the temporary memory
        #     self.cache.store_transition(obs_dict, action, new_obs_dict)
        #     #Preparing for next step
        #     obs_dict = new_obs_dict
        #     self.timestep += 1
                                  
        # # storing in the memory the entire episode
        # self.memory.store(self.cache)
    
    def select_action(self,obs_dict,explore = True):
        with torch.no_grad():
            obs = torch.as_tensor(obs_dict, dtype = torch.float32)
            goal = torch.as_tensor(goal, dtype = torch.float32)  
            action = self.actor(obs,goal).detach().numpy()
            if explore:
                action += self.env_params['action_bound'] * self.noise_eps * np.random.randn(self.env_params['action_dim'])
            action = np.clip(action, -self.env_params['action_bound'], self.env_params['action_bound'])
        return action
    
    def sample(self):
		# Sample replay buffer 
        transitions, weights = self.memory.sample()

        # preprocess
        o, new_o, g = transitions['observation'], transitions['new_observation'], transitions['goal']
        transitions['observation'], transitions['goal'] = self._preproc_og(o, g)
        transitions['new_observation'], transitions['new_goal'] = self._preproc_og(new_o, g)

        obs_norm = self.o_norm.normalize(transitions['observation'])
        g_norm = self.g_norm.normalize(transitions['goal'])
        new_obs_norm = self.o_norm.normalize(transitions['obs_next'])
        new_g_norm = self.g_norm.normalize(transitions['g_next'])
        action = self.to_torch(transitions['action'])
        reward = self.to_torch(transitions['reward'])

        return obs_norm, g_norm, action, reward, new_obs_norm, new_g_norm
    
    def _preproc_og(self, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g
    
    def _preproc_inputs(self, o, g):
        '''State normalization'''

        o_norm = self.o_norm.normalize(o)
        g_norm = self.g_norm.normalize(g)
 
        inputs = np.concatenate([o_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
        return inputs
    
    def learning_step(self): 
        #Sampling of the minibatch
        transitions, weights = self.memory.sample(timestep = self.timestep)
        weights = torch.as_tensor(weights, dtype = torch.float32)
        observations = torch.as_tensor(transitions['observation'], dtype=torch.float32)
        goals = torch.as_tensor(transitions['goal'], dtype=torch.float32)
        new_observations = torch.as_tensor(transitions['new_observation'], dtype=torch.float32)
        actions = torch.as_tensor(transitions['action'], dtype=torch.float32)
        rewards = torch.as_tensor(transitions['reward'], dtype=torch.float32).reshape(-1,1)
            
        #Value Optimization
        with torch.no_grad():
            best_actions = self.target_actor(new_observations, goals) #(batch_size, 1)
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
        priorities = torch.abs(targets - estimations).detach().numpy() + 1e-5
        self.memory.update_priorities(priorities)
        
        #Don't waste computational effort
        for param in self.critic.parameters():
            param.requires_grad = False
                
        #Policy Optimization
        estimated_actions = self.actor(observations,goals)
        policy_loss = -self.critic(observations,goals,estimated_actions).mean()
        # policy_loss += self.action_l2 * (estimated_actions / self.env_params['action_bound']).pow(2).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
                
        #Reactivate computational graph for critic
        for param in self.critic.parameters():
            param.requires_grad = True
            
        return value_loss, policy_loss
    
    def update_target_networks(self, polyak=None):
        polyak = self.polyak if polyak is None else polyak
        with torch.no_grad():
            for target, online in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.data.mul_(polyak)
                target.data.add_((1 - polyak) * online.data)

            for target, online in zip(self.target_actor.parameters(), self.actor.parameters()):
                target.data.mul_(polyak)
                target.data.add_((1 - polyak) * online.data)
                
    def evaluate(self, render = False):
        # starting point
        obs_dict = self.env.reset()[0]
        observation = obs_dict['observation']
        goal = obs_dict['desired_goal'] #never changes in an episode
        info = None     
        # Episode playing
        for t in range(self.env_params['max_steps']):
            action = self.select_action(observation,goal,explore = False)
            new_obs_dict, _, _, _, info = self.env.step(action)
            new_observation = new_obs_dict['observation']
            if render: self.env.render()
            #Preparing for next step
            observation = new_observation
        success = info['is_success']
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
        plt.plot(range(1,success_rate.size+1), success_rate)
        plt.show(block = True)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    def to_torch(self, array, copy=True):
        if copy:
            return torch.tensor(array, dtype=torch.float32).to(device)
        return torch.as_tensor(array).to(device)
    
from enum import Enum
class Mode(Enum):
    TRAIN = 1
    TEST = 2
     
def reach(mode = None):
    if mode == Mode.TRAIN:
        env = gym.make('FetchReach-v2')
        agent = FetchAgent("ddpg_hgr_fetch_reach", env, max_episodes = 500, window = 20)
        agent.train()    
        agent.save() #Done training and saving the model
    if mode == Mode.TEST:
        env = gym.make('FetchReach-v2', render_mode = "human")
        agent = FetchAgent("ddpg_hgr_fetch_reach", env)
        agent.load()
        agent.evaluate(num_ep = 10, render = True)
        
def push(mode = None):
    if mode == Mode.TRAIN:
        env = gym.make('FetchPush-v2')
        agent = FetchAgent("ddpg_hgr_fetch_push", env, max_episodes = 50000, window = 200)
        agent.train()    
        agent.save() #Done training and saving the model
    if mode == Mode.TEST:
        env = gym.make('FetchPush-v2', render_mode = "human")
        agent = FetchAgent("ddpg_hgr_fetch_push", env)
        agent.load()
        agent.evaluate(num_ep = 10, render = True)
        
        
def pickandplace(mode = None):
    if mode == Mode.TRAIN:
        env = gym.make('FetchPickAndPlace-v2')
        agent = FetchAgent("ddpg_hgr_fetch_pickplace", env, max_episodes = 20000, window = 200)
        agent.train()    
        agent.save() #Done training and saving the model
    if mode == Mode.TEST:
        env = gym.make('FetchPickAndPlace-v2', render_mode = "human")
        agent = FetchAgent("ddpg_her_fetch_pickplace", env)
        agent.load()
        agent.evaluate(num_ep = 10, render = True)
            
if __name__ == "__main__":
    reach(Mode.TRAIN)