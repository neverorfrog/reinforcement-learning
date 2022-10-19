import random

import numpy as np
import gym
import time
from gym import spaces
import os


# custom 2d grid world enviroment
class GridWorld(gym.Env):
    metadata = {'render.modes': ['console']}

    # actions available
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


    def __init__(self, width, height):
        super(GridWorld, self).__init__()
        self.ACTION_NAMES = ["UP", "LEFT", "DOWN", "RIGHT"]
        self.num_actions = 4

        self.size = width * height  # size of the grid world
        self.num_states = self.size
        self.width = width
        self.height = height
        self.num_obstacles = int((width+height)/2)
        self.end_state = np.array([height - 1, width - 1], dtype=np.uint8) # goal state = bottom right cell

        # actions of agents : up, down, left and right
        self.action_space = spaces.Discrete(4)
        # observation : cell indices in the grid
        self.observation_space = spaces.MultiDiscrete([self.height, self.width])

        self.obstacles = np.zeros((height, width))

        for i in range(self.num_obstacles):
            self.obstacles[ random.randrange(height) , random.randrange(width)] = 1

        self.num_steps = 0
        self.max_steps = height*width

        self.current_state = np.zeros((2), np.uint8)#init state = [0,0]

        self.directions = np.array([
            [-1,0], #UP
            [0,-1], #LEFT
            [1,0], #DOWN
            [0,1] #RIGHT
        ])

    def transition_function(self, s, a):
        s_prime = s + self.directions[a,:]

        if s_prime[0] < self.height and s_prime[1] < self.width and (s_prime >= 0).all():
            if self.obstacles[s_prime[0], s_prime[1]] == 0 :
                return s_prime

        return s

    def transition_probabilities(self, s, a):
        prob_next_state = np.zeros((self.heigth, self.width))
        s_prime = self.transition_function(s, a)

        prob_next_state[s_prime[0], s_prime[1]] = 1.0

        return prob_next_state#.flatten()

    def reward_function(self,s):
        r = 0
        if (s == self.end_state).all():
            r = 1

        return r

    def termination_condition(self, s):
        done = False
        #done= ???

        done = (s == self.end_state).all() or self.num_steps > self.max_steps

        return done

    def step(self, action):
        s_prime = self.transition_function(self.current_state, action)
        reward = self.reward_function(s_prime)
        done = self.termination_condition(s_prime)

        self.current_state = s_prime
        self.num_steps += 1

        return self.current_state, reward, done, None

    def render(self):
        '''
            render the state
        '''

        row = self.current_state[0]
        col = self.current_state[1]

        for r in range(self.height):
            for c in range(self.width):
                if r == row and c == col:
                    print("| A ", end='')
                elif r == self.end_state[0] and c == self.end_state[1]:
                    print("| G ", end='')
                else:
                    if self.obstacles[r,c] == 1:
                        print('|///', end='')
                    else:
                        print('|___', end='')
            print('|')
        print('\n')

    def reset(self):
        self.current_state = np.zeros((2), np.uint8)
        self.num_steps = 0
        return self.current_state

    def reward_probabilities(self):
        rewards = np.zeros((self.num_states))
        i = 0
        for r in range(self.height):
            for c in range(self.width):
                state = np.array([r,c], dtype=np.uint8)
                rewards[i] = self.reward_function(state)
                i+=1

        return rewards

    def close(self):
        pass
    
    
class NonDeterministicGridWorld(GridWorld):
    def __init__(self, width, height, p=0.8):
        super(NonDeterministicGridWorld, self).__init__(width, height)
        self.probability_right_action = p

    def transition_function(self, s, a):
        s_prime = s + self.directions[a, :]

        #with probability 1 - p diagonal movement
        if random.random() <= 1 - self.probability_right_action:
            if random.random() < 0.5:
                s_prime = s_prime + self.directions[(a+1)%self.num_actions, :]
            else:
                s_prime = s_prime + self.directions[(a-1)%self.num_actions, :]


        if s_prime[0] < self.height and s_prime[1] < self.width and (s_prime >= 0).all():
            if self.obstacles[s_prime[0], s_prime[1]] == 0 :
                return s_prime

        return s

    def transition_probabilities(self, s, a):
        cells = []
        probs = []
        prob_next_state = np.zeros((self.height, self.width))
        s_prime_right =  s + self.directions[a, :]
        if s_prime_right[0] < self.height and s_prime_right[1] < self.width and (s_prime_right >= 0).all():
            if self.obstacles[s_prime_right[0], s_prime_right[1]] == 0 :
                prob_next_state[s_prime_right[0], s_prime_right[1]] = self.probability_right_action
                cells.append(s_prime_right)
                probs.append(self.probability_right_action)

        s_prime = s_prime_right + self.directions[(a + 1) % self.num_actions, :]
        if s_prime[0] < self.height and s_prime[1] < self.width and (s_prime >= 0).all():
            if self.obstacles[s_prime[0], s_prime[1]] == 0 :
                prob_next_state[s_prime[0], s_prime[1]] = (1 - self.probability_right_action) / 2
                cells.append(s_prime.copy())
                probs.append((1 - self.probability_right_action) / 2)

        s_prime = s_prime_right + self.directions[(a - 1) % self.num_actions, :]
        if s_prime[0] < self.height and s_prime[1] < self.width and (s_prime >= 0).all():
            if self.obstacles[s_prime[0], s_prime[1]] == 0 :
                prob_next_state[s_prime[0], s_prime[1]] = (1 - self.probability_right_action) / 2
                cells.append(s_prime.copy())
                probs.append((1 - self.probability_right_action) / 2)

        #normalization
        sump = sum(probs)
        #for cell in cells:
        #    prob_next_state[cell[0], cell[1]] /= sump
        prob_next_state[s[0], s[1]] = 1 - sump
        return prob_next_state
