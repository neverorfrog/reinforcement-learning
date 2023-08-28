from collections import deque
from copy import deepcopy
import numpy as np
import gymnasium as gym
from network import DQN
from common.plotting import ProgressBoard
from common.utils import *
import torch
import torch.nn as nn
from common.buffers import UniformBuffer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepQAgent(HyperParameters):
    def __init__(self, name, env: gym.Env, board: ProgressBoard, 
                 gamma=0.9999, eps=0.999, eps_decay=0.99,
                 max_steps = 200, max_episodes = 500, reward_threshold = 400):
        
        #Hyperparameters
        self.save_hyperparameters()
        
        #Network
        in_dim, out_dim = self.get_dims(env)
        self.Q = DQN(in_dim, out_dim).to(device)
        self.Q_t = deepcopy(self.Q).to(device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr = 0.001)
        self.loss_fn = nn.MSELoss()
        
        #Expereince Replay Buffer
        self.memory =  UniformBuffer(env)
        self.network_update_frequency = 5
        self.target_update_frequency = 20
        self.batch_size = 32
        self.window = 50
        
    
    #Most proabably to be redefined in subclasses    
    def get_dims(self, env: gym.Env):
        in_dim = env.observation_space.n
        out_dim = env.action_space.n
        return in_dim, out_dim
        
    def train(self):
        
        #Life stats
        self.ep = 1
        self.training = True
        self.rewards = deque(maxlen = self.window)
        self.losses = deque(maxlen = self.window)
        
        #Populating the experience replay memory
        self.populate_buffer()
         
        while self.training:
                
            # Episode stats
            steps = 0
            self.ep_reward = 0
            self.ep_mean_loss = 0.

            # ep termination
            done = False
            
            #starting point
            observation = torch.FloatTensor(self.env.reset()[0])
            
            while not done:
                new_observation, done = self.step(observation)

                #Online network update
                if steps % self.network_update_frequency == 0:
                    self.update()
                   
                #Copying online network weights into target network 
                if steps % self.target_update_frequency == 0:
                    self.Q_t.load_state_dict(self.Q.state_dict())
                
                #Termination condition satisfied
                if steps > self.max_steps:
                    done = True
                
                observation = new_observation.detach().clone()
                steps += 1
                
            self.episode_update()
            
            
    def step(self, observation):
        #Choice of action and step in the MDP (eps greedy)
        action = self.epsgreedy(observation)
        new_observation, reward, terminated, truncated, _ = self.env.step(action)
        new_observation = torch.FloatTensor(new_observation)
        done = terminated or truncated
        
        #Storing in the memory
        self.memory.store(observation,action,reward,done,new_observation)
                 
        #stats
        self.ep_reward += reward
        
        return new_observation, done
    
    def epsgreedy(self, observation):
        with torch.no_grad():
            num_actions = self.env.action_space.n
            action_probs = np.ones(num_actions) * self.eps / num_actions
            action_probs[torch.argmax(self.Q(observation))] += (1.0 - self.eps)
            action = np.random.choice(np.arange(num_actions, dtype = int), p = action_probs)
            return action
            
    def update(self): 
        #Sampling and loss function
        minibatch = self.memory.sample(batch_size = self.batch_size)
        loss = self.loss(minibatch)
        self.ep_mean_loss += (1/self.ep)*(loss.item() - self.ep_mean_loss)        
        
        #Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    
    def loss(self, minibatch, weights = None):
        #Transform batch into torch tensors
        rewards = torch.FloatTensor(minibatch.reward).reshape(-1,1).to(device)
        actions = torch.LongTensor(minibatch.action).reshape(-1,1).to(device)
        dones = torch.IntTensor(minibatch.done).reshape(-1,1).to(device)
        observations = torch.stack(minibatch.observation).to(device)
        new_observations = torch.stack(minibatch.new_observation).to(device)
        
        #Estimate q values
        estimations = self.Q(observations) #(batch_size, num_actions)
        estimations = torch.gather(estimations, 1, actions) #(batch_size,1)        
        
        #Target q values (using target network as evaluator and online as estimator)
        with torch.no_grad():
            target_actions = torch.argmax(self.Q(new_observations),1).reshape(-1,1)
            maxqs = torch.gather(self.Q_t(new_observations), 1, target_actions)
            targets = (rewards + (1 - dones) * self.gamma * maxqs)
                    
        #loss function
        loss = self.loss_fn(estimations,targets)

        return loss
    
    def episode_update(self):
        self.eps = max(0.1, self.eps*self.eps_decay)
        self.rewards.append(self.ep_reward)
        self.losses.append(self.ep_mean_loss)
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
                action = self.greedy(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                observation = torch.FloatTensor(observation)
                total_reward += reward
                if render: self.env.render()
                
            if render: print("\tTotal Reward:", total_reward)
            mean_reward = mean_reward + (1/i)*(total_reward - mean_reward)

        if render: print("Mean Reward: ", mean_reward)
        return mean_reward 
    
    def greedy(self, observation):
        with torch.no_grad():
            action_value = self.Q(observation)
            action = torch.argmax(action_value)
            return action.item()
    
    def populate_buffer(self):    
        observation = torch.FloatTensor(self.env.reset()[0])
        for _ in range(10000):
            action = self.epsgreedy(observation)
            new_observation, reward, terminated, truncated, _ = self.env.step(action)
            new_observation = torch.FloatTensor(new_observation)
            done = terminated or truncated
            self.memory.store(observation,action,reward,done,new_observation)
            observation = new_observation.detach().clone()
            if terminated or truncated: 
                observation   = torch.FloatTensor(self.env.reset()[0])
            
    def save(self):
        torch.save(self.network.state_dict(), f"{self.name}.pt")

    def load(self):
        self.network.load_state_dict(torch.load(f"{self.name}.pt"))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret 
  
     
class ControlDeepQAgent(DeepQAgent):
    def __init__(self, name, env: gym.Env, board: ProgressBoard, 
            gamma=0.99, eps=1., eps_decay=0.99):
        super().__init__(name, env, board)  
        
    #Most proabably to be redefined in subclasses    
    def get_dims(self, env: gym.Env):
        in_dim = env.observation_space.shape[0]
        out_dim = env.action_space.n
        return in_dim, out_dim
    
if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode = "rgb_array")     
    # env = gym.make("MountainCar-v0")   
    n = 200
    
    board = ProgressBoard(n, n = max(n / 10, 1))
    agent = ControlDeepQAgent("hi", env, board)
    agent.train()
    
    # testenv = gym.make("MountainCar-v0", render_mode = "human")
    testenv = gym.make('CartPole-v1', render_mode = "human")
    agent.evaluate(testenv, render = True, episodes = 10, max_steps=200)
