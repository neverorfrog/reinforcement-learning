from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import *
import os
import numpy as np
import gymnasium as gym
from networks import *
from networks import *
from plotting import ProgressBoard
import torch
import torch.nn as nn
from buffer import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
np.seterr(all="raise")


class DDPG(Parameters):
    def __init__(self, name, env: gym.Env, board: ProgressBoard = None, window = 50,
                 polyak = 0.95, pi_lr = 0.001, q_lr = 0.001, eps = 1.0, eps_decay = 0.95, 
                 batch_size = 256, gamma=0.99, max_episodes=500):

        # Hyperparameters
        self.save_parameters()
        
        # env params for networks and buffer
        observation = env.reset()[0]
        self.env_params = {'obs_dim': observation['observation'].shape[0], 
                           'action_dim': env.action_space.shape[0], 
                           'action_bound': env.action_space.high[0],
                           'max_steps': env._max_episode_steps}
        # Networks
        self.actor: Actor = StandardActor(self.env_params).to(device)
        self.target_actor: Actor = deepcopy(self.actor).to(device)
        self.critic: Critic = StandardCritic(self.env_params).to(device)
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
        self.start_episodes = 20
        

    def train(self):

        #Life stats
        self.success_rate = []
        self.ep = 0
        self.timestep = 0
        ep_successes = deque(maxlen = self.window)

        # Populating the experience replay memory
        self.populate_buffer()
        
        for self.ep in tqdm(range(1, self.max_episodes)):
            
            # starting point
            obs_dict = self.env.reset()[0]
            observation = obs_dict['observation']

            for t in range(self.env_params['max_steps']):
                action = self.select_action(observation, noise_weight = self.eps)
                new_obs_dict, reward, _, _, _ = self.env.step(action)
                new_observation = new_obs_dict['observation']
                done = t == self.env_params['max_steps'] - 1
                # Storing in the memory
                self.memory.store(observation,action,reward,done,new_observation) 
                # Online network update
                self.learning_step()
                # Copying online network weights into target network
                self.update_target_networks()

                observation = new_observation

            #Logging
            self.ep += 1
            ep_successes.append(self.evaluate())
            mean_success = np.mean(ep_successes)
            self.success_rate.append(mean_success)
            print(f"Episode {self.ep+1} SUCCESS RATE {mean_success:.2f}\n")
            


    def select_action(self, obs, noise_weight = 0.5):
        with torch.no_grad(): 
            action = self.actor(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()
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
            
      
               
    def evaluate(self, render:bool = False):
        obs_dict = self.env.reset()[0]
        observation = torch.FloatTensor(obs_dict['observation'])
        info = None 
        for _ in range(self.env_params['max_steps']):
            action = self.select_action(observation, noise_weight = 0)
            new_obs_dict, _, _, _, info = self.env.step(action)
            new_observation = torch.FloatTensor(new_obs_dict['observation'])
            if render: self.env.render()
            observation = new_observation
            
        success = 1 if info['is_success'] else 0 
        return success  
    
    def populate_buffer(self):   
        for _ in range(self.start_episodes): 
            obs_dict = self.env.reset()[0]
            observation = obs_dict['observation']
            for _ in range(self.start_episodes):
                with torch.no_grad(): 
                    action = self.select_action(observation, noise_weight = 1)
                new_obs_dict, reward, terminated, truncated, _ = self.env.step(action)
                new_observation = new_obs_dict['observation']
                done = terminated or truncated
                self.memory.store(observation,action,reward,done,new_observation)
                observation = new_observation
                observation = self.env.reset()[0]
                observation = obs_dict['observation']
                
            
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
        xaxis = np.arange(start=1,stop=success_rate.size+1) * 50 #* self.env_params['max_steps']
        plt.plot(xaxis, success_rate)
        plt.show(block = True)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
         
SEEDS = [123]    
def launch(env_name = 'FetchReach-v2', prioritized = True):
    for seed in SEEDS:
        set_global_seeds(seed)
        env = gym.make(env_name)
        agent = DDPG(f"DDPG_{env_name}_{seed}", env, max_episodes = 400, window = 100)
        agent.train()    
        agent.save() #Done training and saving the model
 
def test(env_name = 'FetchReach-v2', prioritized = True):
    for seed in SEEDS:
        set_global_seeds(seed)
        env = gym.make(env_name, render_mode = "human")
        agent = DDPG(f"DDPG_{env_name}_{seed}", env)
        agent.plot_success()
        agent.load()
        for _ in range(10):
            agent.evaluate(render = True)
            
if __name__ == "__main__":
    reach = 'FetchReach-v2'
    push = 'FetchPush-v2'
    pickandplace = 'FetchPickAndPlace-v2'
    launch(reach, True)
    # launch(pickandplace, True)
    # launch(pickandplace, False)