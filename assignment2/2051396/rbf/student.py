import random
import numpy as np
import gym
import time
from gym import spaces
import os
import pickle


class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:

    def __init__(self, env): # modify
        self.env = env
        self.tilings = 1
        self.slices = 6
        self.std_dev = [0.15,0.05]
        self.n = (self.slices**2)#*self.tilings #number of features
        self.max = self.env.observation_space.high
        self.min = self.env.observation_space.low
        self.centers = self.centers()
    
    def normalize(self,state):
        return (state - self.min)/(self.max-self.min)

    def centers(self):

        # min_lim = 1/(self.slices+1)
        # max_lim = 1 - min_lim
        min_lim = 0.15
        max_lim = 0.85
         
        c_p = np.linspace(min_lim,max_lim,self.slices)
        c_v = np.linspace(min_lim,max_lim,self.slices)

        # for i in range(self.tilings-1):
        #     c_p_next = np.linspace(min_lim,max_lim,self.slices)
        #     c_v_next = np.linspace(min_lim,max_lim,self.slices)
        #     c_p = np.concatenate((c_p,c_p_next))
        #     c_v = np.concatenate((c_v,c_v_next))

        centers = np.zeros((self.n,self.env.observation_space.shape[0]))
        k = 0
        for i in range(self.slices):
            for j in range(self.slices):
                centers[k] = [c_p[i],c_v[j]]
                k += 1

        return centers
        

    def rbf(self,state,center,sigma):
        """
        input
            state: 2x1 vector (position along x axis and velocity)
            center: 2x1 vector (x and y coordinate of the feature)
            sigma: width of the feature
        ouput
            real number representing the "distance" between state and center
        """
        num = -np.power(np.linalg.norm(state - center),2)
        den = 2*np.power(sigma,2)
        return np.exp(num/den)

        
    def encode(self, state):
        """
        input
            state: 2x1 vector
        output
            n sized vector with features representing the state
        """

        state = self.normalize(state)
        features = np.zeros((self.n))

        # features_per_tiling = (self.slices**2)
        # for i in range(self.tilings):
        #     offset = i*features_per_tiling
        for r in range(self.n):
            features[r] = self.rbf(state, self.centers[r],0.15)

        return features

            
    @property
    def size(self): # modify
        return self.n #n is the number of features

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder,
        alpha=0.01, alpha_decay=1, gamma=0.9999, epsilon=0.3, epsilon_decay=1, lambda_=0.9): # modify if you want
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1,1) #column vector
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s) 
        s_prime_feats = self.feature_encoder.encode(s_prime)

        #update of the eligibility related to the current state,action
        self.traces[action] += s_feats

        #temporal difference error
        td_error = reward + (1-done)*self.gamma*self.Q(s_prime_feats).max() - self.Q(s_feats)[action]

        #update of the weights
        self.weights[action] += self.alpha*td_error*self.traces[action] 

        #update of all the eligibility traces
        self.traces *= self.gamma * self.lambda_

        
    def update_alpha_epsilon(self): # modify
        self.epsilon = max(0.2, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay
        
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None):  # modify
        if epsilon is None: epsilon = self.epsilon
        
        if random.random() < epsilon:
            return self.env.action_space.sample() 
        else:
            return self.policy(state)
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
