import random
import numpy as np
import gymnasium as gym
from utils import make_epsilon_greedy_policy
from plotting import ProgressBoard

def sarsa(env: gym.Env, episodes: int, alpha: float = 0.7, eps: float = 0.1, gamma: float = 0.99, board: ProgressBoard = None):
    # Initializing the policy randomly (Q incarnates the policy right now because we choos the action maximising the action-value)
    if board is None: board = ProgressBoard(episodes)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = make_epsilon_greedy_policy(Q, eps, env.action_space.n)
        
    for episode in range(episodes):
        state = env.reset()[0] #initialize S: state from reset is a tuple
        action = np.argmax(Q[state,:]) #take greedy action
        total_reward = 0
        terminated = False
        
        while not terminated:
            #Step in the MDP based on current policy (ONPOLICY)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            #Estimation of the action-value when selecting an eps-greedy action and then following the same policy afterwards (ON POLICY) 
            # new_action = env.action_space.sample() if (random.random() < eps) else np.argmax(Q[new_state,:])
            new_action = np.random.choice(np.arange(num_actions), p = policy(new_state))
            target = reward + (1-terminated)*gamma*Q[new_state, new_action]
            error = target - Q[state,action]
            Q[state,action] += alpha * error
            
            #Step
            state = new_state
            action = new_action
        
        #Logging
        target_policy = np.argmax(Q, axis = 1)
        testreward = test_policy(env, target_policy, episodes = 1)
        board.draw(episode, testreward, "sarsa")

    policy = np.argmax(Q, axis = 1)
    
    return policy

def qlearning(env: gym.Env, episodes: int, alpha: float = 0.7, eps: float = 0.1, gamma: float = 0.99, board: ProgressBoard = None):
    # Initializing the policy randomly (Q incarnates the policy right now because we choos the action maximising the action-value)
    if board is None: board = ProgressBoard(episodes, n = 1)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = make_epsilon_greedy_policy(Q, eps, env.action_space.n) # behavioral policy
        
    for episode in range(episodes):
        state = env.reset()[0] #initialize S: state from reset is a tuple
        total_reward = 0    
        terminated = False
        
        while not terminated:
            #Behavior Policy: Step in the MDP based on epsilon greedy action (OFFPOLICY)
            action = np.random.choice(np.arange(num_actions), p = policy(state))
            new_state, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            
            #Target Policy: Update rule to learn in just one update rule (like value iteration)            
            target = reward + (1-terminated) * gamma * np.max(Q[new_state])
            error = target - Q[state,action]
            Q[state,action] += alpha * error
            
            #State we are currently in
            state = new_state
                    
        #Logging
        target_policy = np.argmax(Q, axis = 1)
        testreward = test_policy(env, target_policy, episodes = 1)
        board.draw(episode, testreward, "qlearning")
        
    policy = np.argmax(Q, axis = 1)
    
    return policy

def expected_sarsa(env: gym.Env, episodes: int, alpha: float = 0.7, eps: float = 0.1, gamma: float = 0.99, board: ProgressBoard = None):
    # Initializing the policy randomly (Q incarnates the policy right now because we choos the action maximising the action-value)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = make_epsilon_greedy_policy(Q, eps, env.action_space.n)
        
    for episode in range(episodes):
        state = env.reset()[0] #initialize S: state from reset is a tuple
        total_reward = 0    
        terminated = False
        
        while not terminated:
            #Step in the MDP based on epsilon greedy action
            action = np.random.choice(np.arange(num_actions), p = policy(state))
            new_state, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            
            #Update rule 
            expected_Q = policy(new_state) @ Q[new_state]
            target = reward + (1-terminated) * gamma * expected_Q
            error = target - Q[state,action]
            Q[state,action] += alpha * error
            
            #State we are currently in
            state = new_state
                    
        #Logging
        target_policy = np.argmax(Q, axis = 1)
        testreward = test_policy(env, target_policy, episodes = 1)
        board.draw(episode, testreward, "expected_sarsa")
        
    policy = np.argmax(Q, axis = 1)
    
    return policy

def double_qlearning(env: gym.Env, episodes: int, alpha: float = 0.7, eps: float = 0.1, gamma: float = 0.99, board: ProgressBoard = None):
    # Initializing the policy randomly (Q incarnates the policy right now because we choos the action maximising the action-value)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q1 = np.zeros((num_states, num_actions))
    Q2 = np.zeros((num_states, num_actions))
    policy = make_epsilon_greedy_policy(Q1+Q2, eps, env.action_space.n)
        
    for episode in range(episodes):
        state = env.reset()[0] #initialize S: state from reset is a tuple
        total_reward = 0    
        terminated = False
        
        while not terminated:
            #Step in the MDP based on epsilon greedy action 
            action = np.random.choice(np.arange(num_actions), p = policy(state))
            new_state, reward, terminated, _, _ = env.step(action)
            total_reward += reward
            
            #Update rule
            if random.random() < 0.5:        
                target = reward + (1-terminated) * gamma * Q2[new_state, np.argmax(Q1[new_state])]
                error = target - Q1[state,action]
                Q1[state,action] += alpha * error
            else:
                target = reward + (1-terminated) * gamma * Q1[new_state, np.argmax(Q2[new_state])]
                error = target - Q2[state,action]
                Q2[state,action] += alpha * error    
            
            #State we are currently in
            state = new_state
                    
        #Logging
        target_policy = np.argmax(Q1+Q2, axis = 1)
        testreward = test_policy(env, target_policy, episodes = 1)
        board.draw(episode, testreward, "double_qlearning")
        
    policy = np.argmax(Q1+Q2, axis = 1)
    
    return policy


def sarsa_lambda(env, alpha=0.2, gamma=0.99, lambda_= 0.8, initial_epsilon=1.0, n_episodes=5000, board: ProgressBoard = None):

    ####### Hyperparameters
    # alpha = learning rate
    # gamma = discount factor
    # lambda_ = elegibility trace decay
    # initial_epsilon = initial epsilon value
    # n_episodes = number of episodes

    Q = np.random.rand(env.observation_space.n, env.action_space.n)
    policy = make_epsilon_greedy_policy(Q, initial_epsilon, env.action_space.n)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # init epsilon
    epsilon = initial_epsilon
    received_first_reward = False


    for episode in range(n_episodes+1):

        E = np.zeros((env.observation_space.n, env.action_space.n))

        state, _ = env.reset()
        action = np.random.choice(np.arange(num_actions), p = policy(state))
        done = False

        #cycling through the current episode
        while not done:

            #simulate the action
            new_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            
            #update q table and eligibility (backward) for current state and action
            new_action = np.random.choice(np.arange(num_actions), p = policy(state))
            td_error = reward + (1-done)*gamma*Q[new_state,new_action] - Q[state,action]
            E[state,action] += 1
            Q = Q + alpha * td_error * E

            if not received_first_reward and reward > 0:
                received_first_reward = True

            #update of the eligibility traces
            E = E * gamma * lambda_

            # update current state and action
            state = new_state
            action = new_action

        # update current epsilon
        if received_first_reward:
            epsilon = 0.99 * epsilon
        
        #Logging
        target_policy = np.argmax(Q, axis = 1)
        testreward = test_policy(env, target_policy, episodes = 1)
        board.draw(episode, testreward, "qlearning")
            
    policy = np.argmax(Q, axis = 1)
    
    return policy



def test_policy(env, policy, episodes = 5):
    mean_reward = 0.
    for i in range(1, episodes+1):
        state = env.reset()[0]
        terminated = False
        k = 1
        total_reward = 0
        while not terminated and k < 100:
            action = policy[state]
            state, reward, terminated, _, _ = env.step(action)
            k += 1
            total_reward += reward
            
        mean_reward = mean_reward + (1/i)*(total_reward - mean_reward)
    return mean_reward