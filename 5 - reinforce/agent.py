from collections import deque
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    
    def __init__(self, in_dim, out_dim, bias = True):
        super().__init__()
        #Linear layers
        self.activation = nn.ReLU()
        self.net = nn.Sequential(nn.Linear(in_dim,32,bias), nn.Tanh(),
                                 nn.Linear(32,out_dim,bias), nn.Softmax(dim = 0))

    
    def forward(self, x) -> torch.Tensor:
        '''
        Input: a state x
        Output: pmf for the actions
        '''
        return self.net(x)


class REINFORCE():
    def __init__(self, name, env: gym.Env, 
                 eps = 1e-6, gamma=0.999, max_steps = 200, max_episodes = 500, reward_threshold = 400):
        #Network
        self.policy = Policy(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), 0.001)
        
        #Plotting
        self.window = 50
        
        #Lists for loss function
        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        
        
    def sampleAction(self, x, exploration = True):
        '''
        Input: a state x
        Output: a sampled action (in this case the actions are discrete)
        '''
        #Action probabilities
        probs = self.policy(torch.FloatTensor(x))
        # Return Action and LogProb
        action = probs.argmax(-1)
        if exploration:
            distribution = Categorical(probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            self.probs.append(log_prob)
        return action.item()
    
    def train(self):
        
        #Life stats
        self.ep = 1
        self.training = True
        
        while self.training:
                
            # Episode stats
            steps = 0

            # ep termination
            done = False
            
            #starting point
            observation = self.env.reset()[0]
            
            while not done:
                
                #Sampling an action and registering the result of taking it
                action = self.sampleAction(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                self.rewards.append(reward)
                
                #Termination condition
                if steps > self.max_steps:
                    done = True
                done = terminated or truncated
                steps += 1
              
            print(f'\rEpisode {self.ep} Ep_Reward: {np.sum(self.rewards)} \t')
            self.update()
                 
            
    def update(self):
        #Discounted return
        running_return = 0
        discounted_returns = []
        for reward in self.rewards[::-1]:
            running_return = reward + self.gamma*running_return
            discounted_returns.insert(0, running_return)
        
        #Loss computation
        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, discounted_returns):
            exit(log_prob)
            loss += log_prob.mean() * delta * (-1)
                                
        #Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        self.ep += 1 
            
    def save(self):
        torch.save(self.network.state_dict(), f"{self.name}.pt")

    def load(self):
        self.network.load_state_dict(torch.load(f"{self.name}.pt"))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret 
    
    
if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode = "rgb_array")     
    num_ep = 200
    agent = REINFORCE("hi", env)
    agent.train()