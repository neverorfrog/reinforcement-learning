import TD as td
import gymnasium as gym
from common.utils import *
from common.plotting import *

#Training
env = gym.make('CliffWalking-v0')
episodes = 80
board = ProgressBoard(episodes, n = max(episodes / 100, 1))
policy1 = td.qlearning(env, episodes, eps = 0.8, board = board)
policy2 = td.sarsa(env, episodes, eps = 0.8, board = board)
policy3 = td.expected_sarsa(env, episodes, eps = 0.8, board = board)
policy4 = td.double_qlearning(env, episodes, eps = 0.8, board = board)

#Testing
env = gym.make('CliffWalking-v0', render_mode = "human")
test_policy(env, policy1, episodes = 1, max_steps=20)
test_policy(env, policy2, episodes = 1, max_steps=20)
test_policy(env, policy3, episodes = 1, max_steps=20)
test_policy(env, policy4, episodes = 1, max_steps=20)
board.block()


'''
Unknown model
- In dynamic programming we had the model, here we don't
- It means that in the Bellman equation (page 59) we don't have the transition probabilities and the value for s'
- What we do is sampling the return instead of explicitly computing it

How do we sample the return?
- In MONTECARLO we generate the entire episode
    - After that, we iterate over the state-action pairs backwards in time
    - For each state-action pair, we sample its return by summing the discounted rewards until the end of the episode
    - So, for each state-action pair, there is an entry in a tensor of shape (#ep, #states, #actions)
    - We then estimate the value of a state-action pair by averaging along the first dimension of the tensor
- In TEMPORAL DIFFERENCE methods we simply reduce the size of the sampled return, by just bootstrapping one step ahead instead of the entire episode
    - So we don't average anything, the estimated value is just a single formula
'''

