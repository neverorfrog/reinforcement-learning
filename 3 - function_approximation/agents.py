import numpy as np
import gymnasium as gym
from common.plotting import ProgressBoard
from encoder import *
from estimator import Estimator
from common.utils import *
from common.plotting import *

class Agent(HyperParameters):
    def __init__(self, name, env, Q, board: ProgressBoard = None, 
                 alpha=0.005, alpha_decay=0.9999, gamma=0.9999, eps=1., eps_decay=0.99):
        self.save_hyperparameters()
        self.name = name
        self.env = env
        self.Q = Q
        self.board = board
        self.behavior_policy = EpsGreedyPolicy(self.Q, self.eps, env.action_space.n)
        
    def train(self, n_episodes=200, max_steps_per_episode=200):
        for episode in range(n_episodes):
            self.observation = self.env.reset()[0]
            total_reward = 0
            for _ in range(max_steps_per_episode):
                new_observation, reward, terminated, truncated = self.step_update(self.observation)
                total_reward += reward
                self.observation = new_observation
                if terminated or truncated: break
            self.episode_update(episode)
    
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
                action = self.Q.greedy_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if render: self.env.render()
                steps += 1
                
            if render: print("\tTotal Reward:", total_reward)
            mean_reward = mean_reward + (1/i)*(total_reward - mean_reward)

        if render: print("Mean Reward: ", mean_reward)
        return mean_reward    
    
    
class Qlearning(Agent):
    def __init__(self, env, Q: Estimator, board: ProgressBoard = None, alpha=0.005, alpha_decay=0.9999, gamma=0.9999, eps=1., eps_decay=0.99):
        super().__init__("qlearning", env, Q, board, alpha, alpha_decay, gamma, eps, eps_decay)
        self.encoder = RBFFeatureEncoder(env)
        
    def step_update(self, state):
        #Choice of action and step in the MDP
        state_f = self.encoder(state)
        action = self.behavior_policy(state_f)
        new_state, reward, terminated, _, _ = self.env.step(action)      
        new_state_f = self.encoder(new_state)
        
        #TD Update (on th weights)
        td_error = reward + (1 - terminated) * self.gamma * self.Q(new_state_f).max() - self.Q(state_f)[action]
        self.Q.weights[action] += self.alpha*td_error*self.traces[action]
        
        return new_state, reward, terminated

class TDLambda(Agent):
    def __init__(self, env, Q: Estimator, board: ProgressBoard = None, alpha=0.005, alpha_decay=0.9999, gamma=0.9999, eps=1., eps_decay=0.99, lambda_ = 0.9):
        super().__init__("tdlambda", env, Q, board, alpha, alpha_decay, gamma, eps, eps_decay)
        self.encoder = RBFFeatureEncoder(env)
        self.shape = (self.env.action_space.n, self.encoder.size)
        self.traces = np.zeros(self.shape)
        self.lambda_ = lambda_
        
    def step_update(self, state):
        #Choice of action and step in the MDP
        state_f = self.encoder(state)
        action = self.behavior_policy(state_f)
        new_state, reward, terminated, _, _ = self.env.step(action)      
        new_state_f = self.encoder(new_state)

        #update of the traces relative to the current action, yields e_t
        self.traces[action] += state_f

        #update of the weights, yields w_t+1
        td_error = reward + (1-terminated)*self.gamma*self.Q(new_state_f).max() - self.Q(state_f)[action]
        self.Q.weights[action] += self.alpha*td_error*self.traces[action]

        #update of the eligibility traces, yields part of e_t+1
        self.traces *= self.gamma * self.lambda_
        
        return new_state, reward, terminated
    

if __name__ == "__main__":
    # env = gym.make('CliffWalking-v0')
    env = gym.make("MountainCar-v0", render_mode = "rgb_array")
    n_episodes = 90
    board = ProgressBoard(n_episodes, n = max(n_episodes / 100, 1))
    encoder = RBFFeatureEncoder(env)
    Q = Estimator(env, encoder)
    agent = TDLambda(env, Q, board)
    agent.train(n_episodes)
    
    testenv = gym.make("MountainCar-v0", render_mode = "human")
    agent.evaluate(testenv, render = True, episodes = 2, max_steps=200)
        

        

