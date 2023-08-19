import numpy as np
import inspect
import torch

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
            
class EpsGreedyPolicy:
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon."""
    def __init__(self, Q, eps, num_actions):
        self.Q = Q
        self.eps = eps
        self.num_actions = num_actions
        
    def action_probs(self, state):
        action_p = np.ones(self.num_actions, dtype=float) * self.eps / self.num_actions
        with torch.no_grad(): best_action = torch.argmax(self.Q(state))
        action_p[best_action] += (1.0 - self.eps)
        return action_p
        
    def __call__(self, state):
        probs = self.action_probs(state)
        actions = np.arange(self.num_actions, dtype = int)
        return np.random.choice(actions, p = probs)

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def test_policy(env, policy, render:bool = True, episodes = 5, max_steps: int = 100):
    
    mean_reward = 0.
    
    for i in range(1, episodes+1):
        if render: print(f"Starting game {i}")

        state = env.reset()[0]
        
        terminated = False
        total_reward = 0
        steps = 0
        
        while not terminated and steps < max_steps:
            action = policy[state]
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if render: env.render()
            steps += 1
            
        if render: print("\tTotal Reward:", total_reward)
        mean_reward = mean_reward + (1/i)*(total_reward - mean_reward)

    if render: print("Mean Reward: ", mean_reward)
    return mean_reward