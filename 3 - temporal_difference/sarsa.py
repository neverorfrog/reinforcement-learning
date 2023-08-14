import random
import numpy as np
from tools.plotting import *
import gymnasium as gym


def sarsa(env: gym.Env, episodes: int, alpha: float = 0.1, eps: float = 0.2, gamma: float = 0.99):
    # Initializing the policy randomly (Q incarnates the policy right now because we choos the action maximising the action-value)
    board = ProgressBoard(episodes)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    action_values = np.zeros((num_states, num_actions))
        
    for episode in range(episodes):
        new_state = env.reset()[0] #initialize S: state from reset is a tuple
        new_action = np.argmax(action_values[new_state,:]) #take greedy action
        total_reward = 0

        terminated = False
        while not terminated:
            #State we are currently in
            state = new_state
            action = new_action
            
            #Step in the MDP
            new_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            #Estimation of the action-value when selecting an eps-greedy action and then following the same policy afterwards (ON POLICY) 
            new_action = eps_greedy(eps, action_values, new_state)            
            old_estimate = action_values[state,action]
            target = reward + gamma*action_values[new_state, new_action]
            action_values[state,action] = old_estimate + alpha*(target - old_estimate)
        
        board.draw(episode, total_reward, "reward", every_n = 10)
        
    policy = np.argmax(action_values, axis = 1)
    
    return policy, action_values

def eps_greedy(eps, action_values, state):
    if random.random() > eps:
        return np.argmax(action_values[state,:]) 
    else:
        return random.randint(0,3)