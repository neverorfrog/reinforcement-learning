import random
import numpy as np


def sarsa(env, epochs, alpha = 0.01, eps = 0.5, gamma = 0.99):
    # Initializing the policy randomly (Q incarnates the policy right now because we choos the action maximising the action-value)
    action_values = np.random.rand(env.num_states, env.num_actions)
    action_values[-1,:] = np.zeros((1, env.num_actions))
        
    for epoch in range(epochs):
        new_state = env.state2index(env.reset()) #initialize S
        new_action = np.argmax(action_values[new_state,:]) #take greedy action

        terminated = False
        while not terminated:
            #State we are currently in
            state = new_state
            action = new_action
            
            #Step in the MDP
            new_state, reward, terminated, _ = env.step(action)
            new_state = env.state2index(new_state)
            
            #Estimation of the action-value when selecting an eps-greedy action and then following the same policy afterwards (ON POLICY) 
            new_action = eps_greedy(eps, action_values, new_state)
            old_estimate = action_values[state,action]
            target = reward + gamma*action_values[new_state, new_action]
            action_values[state,action] = old_estimate + alpha*(target - old_estimate)
    
    policy = np.argmax(action_values, axis = 1)
    policy = policy.reshape(env.height, env.width)
    
    return policy, action_values

def eps_greedy(eps, action_values, state):
    if random.random() < eps:
        return np.argmax(action_values[state,:]) 
    else:
        return random.randint(0,3)