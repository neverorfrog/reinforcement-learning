import numpy as np
import gymnasium as gym
from tools.plotting import ProgressBoard
from encoder import *
from tools.agent import *

    
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
        

        

