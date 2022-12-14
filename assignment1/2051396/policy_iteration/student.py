import random

import numpy as np
import gym
import time
from gym import spaces
import os


def value_iteration(env):
    gamma = 0.99
    iters = 100

    # initialize values
    values = np.zeros((env.num_states))
    best_actions = np.zeros((env.num_states), dtype=int)
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1

    for i in range(iters):
        v_old = values.copy()
        for s in range(env.num_states):
            state = STATES[s]

            if (state == env.end_state).all() or i >= env.max_steps:
                continue  # if we reach the termination condition, we cannot perform any action

            max_va = -np.inf
            best_a = 0
            for a in range(env.num_actions):

                next_state_prob = env.transition_probabilities(state, a).flatten()
                va = (next_state_prob*(REWARDS + gamma*v_old)).sum()

                if va > max_va:
                    max_va = va
                    best_a = a
            values[s] = max_va
            best_actions[s] = best_a
    
    print(i)

    return best_actions.reshape((env.height, env.width))


def policy_iteration(env, gamma=0.99, iters=100):

    # Initializing the states
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1

    # Initializing the policy
    policy = np.zeros(env.num_states, dtype=np.int)
    values = np.zeros(env.num_states, dtype=np.float32)
    

    # Looping for iters iterations to find the optimal policy
    for i in range(iters):

        #Termination condition
        policy_stable = True

        # Looping on all states for policy evaluation until convergence
        eps = 0.1

        while True:

            v_old = values.copy()
            delta = 0

            # Compute the values of the states in according to the current policy
            for s in range(env.num_states):
                # Current state
                state = STATES[s]

                if (state == env.end_state).all() or i >= env.max_steps:
                    continue  # if we reach the termination condition, we cannot perform any action

                next_state_prob = env.transition_probabilities(state, policy[s]).flatten()
                values[s] = (next_state_prob*(REWARDS + gamma*v_old)).sum()

                delta = max(delta, abs(values[s]-v_old[s]))

            if delta < eps:
                break

        # Looping on all states for policy improvement
        for s in range(env.num_states):

            # Current state
            state = STATES[s]

            if (state == env.end_state).all() or i >= env.max_steps:
                continue  # if we reach the termination condition, we cannot perform any action

            # Finding the action that maximises the value for the current state
            max_va = values[s]
            best_action = policy[s]

            for a in range(env.num_actions):

                next_state_prob = env.transition_probabilities(state, a).flatten()
                va = (next_state_prob*(REWARDS + gamma*values)).sum()

                if va > max_va:
                    best_action = a
                    max_va = va

            if policy[s] != best_action:  # checking for termination
                policy_stable = False
                policy[s] = best_action  # policy improvement

        if policy_stable == True:
            print("Finished")
            print(i)
            break

    return policy.reshape(env.height, env.width)
    

    
