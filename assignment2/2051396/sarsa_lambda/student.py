import numpy as np
import random

def epsilon_greedy_action(env, Q, state, epsilon):

    if random.random() < epsilon:
        return env.action_space.sample() 
    else:
        return np.argmax(Q[state])


def sarsa_lambda(env, alpha=0.2, gamma=0.99, lambda_= 0.8, initial_epsilon=1.0, n_episodes=5000):

    ####### Hyperparameters
    # alpha = learning rate
    # gamma = discount factor
    # lambda_ = elegibility trace decay
    # initial_epsilon = initial epsilon value
    # n_episodes = number of episodes

    Q = np.random.rand(env.observation_space.n, env.action_space.n)

    # init epsilon
    epsilon = initial_epsilon
    received_first_reward = False

    #evaluation
    window_dim = 1000
    window = 0
    victories = 0

    print("TRAINING STARTED")
    print("...")

    for ep in range(n_episodes+1):

        E = np.zeros((env.observation_space.n, env.action_space.n))
        
        if ep % window_dim == 0 and ep != 0:
            window += 1
            print("\tWindow {} with success rate {}".format(window, victories/window_dim))
            victories = 0

        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)
        done = False

        #cycling through the current episode
        while not done:

            #simulate the action
            next_state, reward, done, info, _ = env.step(action)
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)

            #update q table and eligibility (backward) for current state and action
            td_error = reward + (1-done)*gamma*Q[next_state,next_action] - Q[state,action]
            E[state,action] += 1

            #update of the q table
            Q = Q + alpha * td_error * E

            if not received_first_reward and reward > 0:
                received_first_reward = True
                print("Received first reward at episode ", ep)

            #update of the eligibility traces
            E = E * gamma * lambda_

            # update current state and action
            state = next_state
            action = next_action

        # update current epsilon
        if received_first_reward:
            epsilon = 0.99 * epsilon
            
        if reward > 0:
            victories += 1

    print("TRAINING FINISHED")
    return Q