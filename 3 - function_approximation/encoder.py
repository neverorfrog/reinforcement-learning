import numpy as np
from common.utils import *

class RBFFeatureEncoder:
    def __init__(self, env): # modify
        self.env = env
        self.tilings = 1
        self.slices = 7
        self.std_dev = [1/(self.slices-1)]
        self.n = (self.slices**2)*self.tilings #number of features

        self.centers = self.centers()
    
    def normalize(self,state):
        result = np.zeros(2)
        max = self.env.observation_space.high
        min = self.env.observation_space.low
        return (state - min)/(max - min)

    def centers(self):

        min_lim = 1/(self.slices+1)
        max_lim = 1 - min_lim
         
        c_p = np.linspace(min_lim,max_lim,self.slices)
        c_v = np.linspace(min_lim,max_lim,self.slices)

        for i in range(self.tilings-1):
            c_p_next = np.linspace(min_lim,max_lim,self.slices)
            c_v_next = np.linspace(min_lim,max_lim,self.slices)
            c_p = np.concatenate((c_p,c_p_next))
            c_v = np.concatenate((c_v,c_v_next))

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
        num = np.power(np.linalg.norm(state - center),2)
        den = 2*np.power(sigma,2)
        return np.exp(-num/den)

        
    def __call__(self, state):
        """
        input
            state: 2x1 vector
        output
            n sized vector with features representing the state
        """

        state = self.normalize(state)
        features = np.zeros((self.n))

        features_per_tiling = (self.slices**2)
        for i in range(self.tilings):
            offset = i*features_per_tiling
            for r in range(features_per_tiling):
                features[r + offset] = self.rbf(state, self.centers[r],self.std_dev[i])

        return features

            
    @property
    def size(self): # modify
        return self.n #n is the number of features
    

