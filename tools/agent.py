from tools.utils import *
from tools.plotting import *

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