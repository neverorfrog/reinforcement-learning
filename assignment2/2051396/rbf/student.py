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
        self.n = 16 #number of features
        self.max = self.env.observation_space.high
        self.min = self.env.observation_space.low
        self.width = (self.max - self.min) / self.n
        self.centers = self.centers()
        # create the rbf encoder
    
    def normalize(self,state):
        return (state - self.min)/(self.max-self.min)


    def centers(self):
        n_states = self.env.observation_space.shape[0] 
        n_slices = int(np.sqrt(self.n))
        n_features = self.n

        c_p = np.linspace(0.2,0.8,n_slices)
        c_v = np.linspace(0.2,0.8,n_slices)
        
        centers = np.zeros((n_features,n_states))
        k = 0
        for i in range(n_slices):
            for j in range(n_slices):
                centers[k] = [c_p[i],c_v[j]]
                k += 1

        return centers
        

    def rbf(self,state,center,sigma):
        """
        state: 2x1 vector (position along x axis and velocity)
        center: 2x1 vector (x and y coordinate of the feature)
        sigma: width of the feature
        """
        #state = state.reshape(-1,1)
        #center = center.reshape(-1,1)
        num = -np.power(np.linalg.norm(state - center),2)
        den = 2*np.power(sigma,2)
        return np.exp(num/den)

        
    def encode(self, state): # modify

        state = self.normalize(state)

        n_states = self.env.observation_space.shape[0]
        n_features = self.n

        features = np.zeros((n_features))

        for r in range(n_features):
            features[r] = self.rbf(state, self.centers[r] , 0.2)

        return features

            
    @property
    def size(self): # modify
        # return the correct size of the observation
        return self.n #n é il numero di features
        #return self.env.observation_space.shape[0]

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder,
        alpha=0.01, alpha_decay=0.99, gamma=0.9999, epsilon=0.3, epsilon_decay=0.9999, lambda_=0.8): # modify if you want
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)#3x20
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
        Q = self.weights@feats
        return Q 
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s) #100x1
        s_prime_feats = self.feature_encoder.encode(s_prime)

        td_error = reward
        if not done:
            td_error += self.gamma*self.Q(s_prime_feats).max()
            
        td_error -=  self.Q(s_feats)[action]#1
                    
        self.traces *= self.gamma * self.lambda_
        self.traces[action] = self.traces[action] + s_feats
        delta_w = td_error*self.traces[action] 
        
        self.weights[action] += self.alpha*delta_w #1x100

        
    def update_alpha_epsilon(self): # modify
        self.epsilon = max(0.2, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay
        pass
        
        
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
