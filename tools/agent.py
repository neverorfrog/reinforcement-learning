import numpy as np
import gymnasium as gym
from tools.encoders import *
import pickle
from tools.plotting import ProgressBoard

class EpsGreedyPolicy:
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon."""
    def __init__(self, Q, eps, num_actions):
        self.Q = Q
        self.eps = eps
        self.num_actions = num_actions
        
    def action_probs(self, state):
        action_p = np.ones(self.num_actions, dtype=float) * self.eps / self.num_actions
        best_action = np.argmax(self.Q(state))
        action_p[best_action] += (1.0 - self.eps)
        return action_p
        
    def __call__(self, state):
        probs = self.action_probs(state)
        actions = np.arange(self.num_actions, dtype = int)
        return np.random.choice(actions, p = probs)
    
class Estimator:
    '''
    Given a state this class spits out th predicted Q function
    This is done by a linear function (state encoded first?) or a neural network
    This base class implements the linear fashion
    '''
    def __init__(self, env: gym.Env, encoder: FeatureEncoder):
        self.env = env
        self.shape = (self.env.action_space.n, encoder.size)
        self.weights = np.random.rand(self.env.action_space.n, encoder.size)
        
    def __call__(self, feats):
        '''State is to be considered already featurized'''  
        feats = feats.reshape(-1,1) #column vector
        return self.weights @ feats


class Agent:
    def __init__(self, name, env, Q: Estimator, board: ProgressBoard = None, alpha=0.005, alpha_decay=0.9999, gamma=0.9999, eps=1., eps_decay=0.99):
        self.env = env
        self.Q = Q
        
        #Hyperparameters
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        
        self.behavior_policy = EpsGreedyPolicy(self.Q, self.eps, env.action_space.n)
        self.board = board
        self.name = name
        
    def train(self, n_episodes=200, max_steps_per_episode=200):
        for episode in range(n_episodes):
            state = self.env.reset()[0]
            total_reward = 0
            for _ in range(max_steps_per_episode):
                new_state, reward, terminated = self.step_update(state)
                total_reward += reward
                state = new_state
                if terminated: break
            self.episode_update(episode)
            
    def policy(self, state):
        if self.encoder is not None: features = self.encoder(state)
        return self.Q(features).argmax()
    
    def episode_update(self, episode):
        self.eps = max(0.2, self.eps*self.eps_decay)
        self.alpha = self.alpha*self.alpha_decay
        testreward = self.evaluate(episodes = 1, render = False)
        self.board.draw(episode, testreward, self.name)
        
        
    def evaluate(self, env = None, render:bool = True, episodes = 3, max_steps: int = 120):
        mean_reward = 0.
        if env is None: env = self.env
        
        for i in range(1, episodes+1):
            if render: print(f"Starting game {i}")

            state = env.reset()[0]
            
            terminated = False
            total_reward = 0
            steps = 0
            
            while not terminated and steps < max_steps:
                action = self.policy(state)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if render: self.env.render()
                steps += 1
                
            if render: print("\tTotal Reward:", total_reward)
            mean_reward = mean_reward + (1/i)*(total_reward - mean_reward)

        if render: print("Mean Reward: ", mean_reward)
        return mean_reward
    
    def save(self):
        with open(self.name, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls,name):
        return pickle.load(open(name,'rb'))