import random
import torch
import numpy as np
device = torch.device("cpu")
import math
from utils import *
np.seterr(all="raise")


class ReplayBuffer:
    def __init__(self, env_params, max_timesteps = None, replay_strategy = 'future', replay_k = 4, reward_function = None, 
                 capacity = 1e6, alpha = 0.6, beta = 0.4):
                
        #Stuff for replaying transitions with a different goal
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_function = reward_function

        #Stuff for keeping track of size (refers to counting episodes, not transitions)
        self.env_params = env_params
        self.T = env_params['max_steps']
        self.capacity_ep = int(capacity // self.T)
        self.size_ep = 0
        self.index_ep = 0
        self.capacity_t = capacity
        self.size_t = 0

        #Implementing here a flattened buffer 
        self.buffer = {'observation': np.empty([self.capacity_ep, self.T, env_params['obs_dim']]),
                       'action': np.empty([self.capacity_ep, self.T, env_params['action_dim']]),
                       'reward': np.empty([self.capacity_ep, self.T]),
                       'new_observation': np.empty([self.capacity_ep, self.T, env_params['obs_dim']]),
                       'goal': np.empty([self.capacity_ep, self.T, env_params['goal_dim']]),
                       'achieved_goal': np.empty([self.capacity_ep, self.T, env_params['goal_dim']]),
                       'new_achieved_goal': np.empty([self.capacity_ep, self.T, env_params['goal_dim']])}
        
        #Stuff for prioritizing
        self.tree = SegmentTree(self.capacity_ep)
        self.max_priority = 1.
        self.priorities_dim = int(((self.T-1) * self.T) / 2)
        self.priorities = np.empty([self.capacity_ep, self.priorities_dim])
        self.td_errors = np.empty([self.capacity_ep, self.priorities_dim])
        self.timestep_couples = np.empty([self.capacity_ep,2],dtype=int)
        index = 0
        for i in range(self.T):
            for j in range(i+1, self.T):
                self.timestep_couples[index] = [i,j]
                index += 1
        self.alpha = alpha
        self.alpha_prime = alpha
        self.beta = LinearSchedule(max_timesteps, 1.0, beta)
        self.beta_prime = beta


    def store(self,episode):
        observations, actions, desired, achieved, new_observations, new_achieved_goals = episode.unpack()
        self.buffer['observation'][self.index_ep] = observations
        self.buffer['action'][self.index_ep] = actions
        self.buffer['new_observation'][self.index_ep] = new_observations
        self.buffer['goal'][self.index_ep] = desired
        self.buffer['achieved_goal'][self.index_ep] = achieved
        self.buffer['new_achieved_goal'][self.index_ep] = new_achieved_goals
        #initializing priority of stored episode at 1
        self.tree.update(self.index_ep, self.max_priority ** self.alpha)
        #initializing priority of contained transitions and the relative td errors at 1
        self.priorities[self.index_ep] = np.ones((self.priorities_dim)) * self.max_priority ** self.alpha_prime
        self.td_errors[self.index_ep] = np.ones((self.priorities_dim)) * self.max_priority
        #size and index update
        self.size_ep = min(self.capacity_ep, self.size_ep + 1)
        self.index_ep = (self.index_ep + 1) % self.capacity_ep

    def sample(self, batch_size=256, timestep = None):
        #empty minibatch
        ep_indices = np.empty([batch_size], dtype=int)
        t_indices = np.empty([batch_size], dtype=int)
        t_now = np.empty([batch_size], dtype=int)
        t_future = np.empty([batch_size], dtype=int)
        weights_t = np.empty([batch_size])
        weights_ep = np.empty([batch_size])
        
        #sampling episodes
        segment = self.tree.sum / batch_size
        for i in range(batch_size):
            sample = random.uniform(segment * i, segment * (i + 1))
            ep_indices[i] = self.tree.get(sample)
        #weighted importance sampling for episodes
        min_prob = self.tree.min / self.tree.sum 
        max_weight_ep = (min_prob * self.size_ep) ** (-self.beta(timestep)) #max w_i for weighted importance sampling
        probs_ep = self.tree.sumtree[ep_indices + self.capacity_ep - 1] / self.tree.sum
        weights_ep = ((probs_ep * self.size_ep) ** (-self.beta(timestep))) / max_weight_ep
                 
        #sampling transitions (and goals)
        priority_sums = np.sum(self.priorities[ep_indices], axis = 1).reshape(-1,1) #reshape to allow broadcasting
        probs_t = np.divide(self.priorities[ep_indices],priority_sums) 
        t_indices = torch.multinomial(torch.as_tensor(probs_t), num_samples=1).squeeze()
        t_now = self.timestep_couples[t_indices][:,0]
        t_future = self.timestep_couples[t_indices][:,1]
        #weighted importance sampling for transitions
        max_weights_t = (np.min(self.priorities[ep_indices], axis = 1) * self.priorities_dim) ** (-self.beta_prime)
        weights_t = (self.priorities[ep_indices,t_indices] * self.priorities_dim) ** (-self.beta_prime)
        weights_t = np.divide(weights_t, max_weights_t)
        
        #total weights
        weights = weights_ep * weights_t
        weights = weights / weights.max()
            
        #extracting the transitions
        transitions = {key: self.buffer[key][ep_indices,t_now].copy() for key in self.buffer.keys()}   
        #replacing some goals with future ones with HER sampling
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_goals = self.buffer['achieved_goal'][ep_indices[her_indices],t_future[her_indices]]
        transitions['goal'][her_indices] = future_goals
        #recompute rewards of changed transitions so that the new achieved goal maybe meets the changed goal
        transitions['reward'] = [self.reward_function(ag,g,None) for ag,g in zip(transitions['new_achieved_goal'],transitions['goal'])]
        
        #for next sampling
        self.last_sampled_episodes = ep_indices
        self.last_sampled_transitions = t_indices
        
        return transitions, weights
    
    def update_priorities(self,priorities):
        ep_indices = self.last_sampled_episodes
        t_indices = self.last_sampled_transitions
        priorities = priorities.squeeze()
        self.max_priority = max(self.max_priority,max(priorities))
        #Update weights of single transitions
        self.priorities[ep_indices,t_indices] = priorities ** self.alpha_prime
        self.td_errors[ep_indices,t_indices] = priorities
        #Update weights of whole episodes
        ep_priorities = np.mean(self.td_errors[ep_indices],axis = 1) ** self.alpha
        for ep_index,ep_priority in zip(ep_indices,ep_priorities):
            self.tree.update(ep_index, ep_priority)
            
    def sample_uniformly(self, batch_size=256):
        #select episodes and timesteps to replay
        max_ep_index = min(self.size_ep, self.capacity_ep - 1)
        ep_indices = np.random.randint(0, max_ep_index, size=batch_size)
        t_indices = np.random.randint(self.T, size=batch_size)
        transitions = {key: self.buffer[key][ep_indices,t_indices].copy() for key in self.buffer.keys()}
        #indices from HER sampling
        indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = (np.random.uniform(size=batch_size) * (self.T - t_indices)).astype(int)
        future_timesteps = (t_indices + future_offset)[indices]
        #replace goal with future achieved goal
        future_goals = self.buffer['achieved_goal'][ep_indices[indices],future_timesteps]
        transitions['goal'][indices] = future_goals
        #recompute rewards of changed transitions so that the new achieved goal maybe meets the changed goal
        transitions['reward'] = [self.reward_function(ag,g,None) for ag,g in zip(transitions['new_achieved_goal'],transitions['goal'])]       
        self.last_sampled_episodes = ep_indices
        self.last_sampled_transitions = t_indices
        return transitions
    
      
class ReplayCache():
    '''Cache of episode during training'''

    def __init__(self, env_params):
        self.T = env_params['max_steps']
        self.obs_dim = env_params['obs_dim']
        self.action_dim = env_params['action_dim']
        self.goal_dim = env_params['goal_dim']
        self.reset()

    def reset(self):
        self.observations = np.empty([self.T, self.obs_dim])
        self.new_observations = np.empty([self.T, self.obs_dim])
        self.actions = np.empty([self.T, self.action_dim])
        self.achieved_goals = np.empty([self.T, self.goal_dim])
        self.new_achieved_goals = np.empty([self.T, self.goal_dim])
        self.desired_goals = np.empty([self.T, self.goal_dim])
    
    def store_transition(self, t, obs_dict, action, new_obs_dict):
        assert(t < self.T)
        self.observations[t] = obs_dict['observation']
        self.achieved_goals[t] = obs_dict['achieved_goal']
        self.desired_goals[t] = obs_dict['desired_goal']
        self.actions[t] = action
        self.new_observations[t] = new_obs_dict['observation']
        self.new_achieved_goals[t] = new_obs_dict['achieved_goal']
        self.is_full = t == self.T - 1

    def unpack(self):
        assert(self.is_full)
        self.is_full = False
        return (self.observations, self.actions, self.desired_goals, self.achieved_goals, self.new_observations, self.new_achieved_goals)

class SegmentTree:
    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity
        self.capacity = (2 * buffer_capacity - 1)
        self.sumtree = np.zeros((self.capacity)) #nodes of the whole tree  
        self.mintree = np.full((self.capacity), float('inf')) #nodes of the whole tree  
        
    def _propagate(self, leaf_index):
        index = leaf_index
        while index > 0: #until i am at the root
            index = math.floor((index - 1)/2) #my parent
            left = 2*index + 1
            right = 2*index + 2
            self.sumtree[index] = self.sumtree[left] + self.sumtree[right]
            self.mintree[index] = min(self.mintree[left],self.mintree[right])
    
    def update(self, transition_index, priority):
        leaf_index = self.buffer2tree(transition_index)
        self.sumtree[leaf_index] = priority
        self.mintree[leaf_index] = priority
        self._propagate(leaf_index)   
        
    def get(self, sample):
        '''
        Given a sample, i.e. a value between 0 and root
        Returns a transition index
        '''
        index = 0
        while index < self.buffer_capacity - 1:
            left = 2*index + 1
            right = 2*index + 2
            if sample <= self.sumtree[left]:
                index = left
            else:
                sample -= self.sumtree[left]
                index = right
         
        return self.tree2buffer(index)
    
    def tree2buffer(self, index):
        return index - self.buffer_capacity + 1
    def buffer2tree(self, index):
        return index + self.buffer_capacity - 1
    
    @property
    def min(self):
        return self.mintree[0]
    @property
    def sum(self):
        return self.sumtree[0]
                
    def __repr__(self):
        return f"SumTree: {self.sumtree} \n MinTree: {self.mintree}"
